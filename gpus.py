#!/usr/bin/env python
# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

"""
Functions to check servers for free GPUs and start experiment on those GPUs.
See example file for usage.
Call this file from command line to search servers for free GPUs and print the result.
"""
import json
import re
import subprocess
import re
import argparse
from plumbum import SshMachine
from plumbum.cmd import sed, awk, git
import plumbum
import csv
from plumbum import colors
import time
from git import Repo
import signal
import sys
from collections import defaultdict
from datetime import datetime

parser = argparse.ArgumentParser(description='Check GPU usage')
parser.add_argument('--verbose', action='store_true',
                    default=False,
                    help='print full information')
parser.add_argument('--ignore', type=str, nargs='+',
                    default=[],
                    help='servers to ignore')


color_free = colors.green
color_me = colors.DodgerBlue1
color_other = colors.red
color_other_light = colors.Orange1


def timeout_handler(signum, frame):
  raise Exception("timeout")

def timeout_call(fn, *args, timeout=10):
  signal.signal(signal.SIGALRM, timeout_handler)
  signal.alarm(timeout)
  try:
    result = fn(args)
    success = True
  except Exception as e:
    result = str(e)
    success = False
  signal.alarm(0)
  return result, success

def check_cpu_usage(remote):
  num_cpus = float(remote["grep"]["-c", "^processor", "/proc/cpuinfo"]())
  top_output = remote["top"]["-b", "-n1"]() 
  lines = top_output.splitlines()
  loadavg = float(lines[0].split()[-1])
  cpu_usage = defaultdict(lambda: 0.0)
  for line in lines[7:]:
    split = line.split()
    user = split[1]
    cpu_pct = float(split[8])
    cpu_usage[user] += cpu_pct/(100 * num_cpus)
  return cpu_usage, loadavg, num_cpus


def inspect_gpus(servers,
                 own_username=None,
                 verbose=False,
                 needed_gpus=-1,
                 memory_threshold=1200,
                 gpu_util_threshold=5,
                 allow_lightly_used_gpus=True,
                 share_with=[],
                 upper_memory_threshold=3000,
                 upper_gpu_util_threshold=30):
    """
    Scan servers for free GPUs, print availability and return a list of free GPUs that can used to
    start jobs on them.

    Requirements:
        ~/.ssh/config needs to be set up so that connecting via `ssh <server>` works. Fos OSX,
        an entry can look like this:

        Host mulga
            User maxigl
            HostName mulga.cs.ox.ac.uk
            BatchMode yes
            ForwardAgent yes
            StrictHostKeyChecking no
            AddKeysToAgent yes
            UseKeychain yes
            IdentityFile ~/.ssh/id_rsa

    Args:
        verbose (bool):           If True, also print who is using the GPUs
        server (list of strings): List of servers to scan


        memory_threshold (int):
        gpu_util_threshold (int): When used memory < lower_memory_threshold and
                                  GPU utilisation < lower_gpu_util_threshold,
                                  then the GPU is regarded as free.

        allow_lightly_used_gpus (bool):
        share_with (list of strings):
        upper_memory_threshold (int):
        upper_gpu_util_threshold (int): If `allow_lightly_used_gpus=True` and memory and gpu
                                        utilisation are under the upper thresholds and there
                                        is so far only one process executed on that GPU who's
                                        user is in in the list `share_with`, then the GPU will
                                        be added to the list of GPUs that can be used to start jobs.

    Return:
        free_gpus: List of dictionaries, each containing the following keys:
                   'server': Name of the server
                   'gpu_nr': Number of the free GPU
                   'double': Whether someone is already using that GPU but it's still considered
                             usuable (see `allow_lightly_used_gpus`)


    """
    # print("GPU color codes: " +
    #       (color_free | "Free" + " | ") +
    #       (color_me | "Own" + " | ") +
    #       (color_other | "Other" + " | ") +
    #       (color_other_light | "Other (light)"))

    all_free_gpus = []
    server_id = 0

    while ((needed_gpus < 0 and server_id < len(servers)) or
            len(all_free_gpus) < needed_gpus):

        server = servers[server_id]
        server_id += 1

        sys.stdout.flush()
        print("{:7}: ".format(server), end='')
        try:
            remote = SshMachine(server)
        except (plumbum.machines.session.SSHCommsError, ValueError, Exception) as e:
            print("ssh fail - maybe server not in .ssh/config?")
            if verbose:
              print("\t(Exception) ", e)
            continue
        try:
          cpu_usage, loadavg, num_cpus = check_cpu_usage(remote)
        except Exception as error:
          print("error for machine:", error)
          continue
        load_ratio = loadavg/num_cpus
        if load_ratio < 0.7:
          cpu_color = colors.green
        elif load_ratio < 1.0:
          cpu_color = colors.yellow
        elif load_ratio < 2.0:
          cpu_color = colors.orange3
        else:
          cpu_color = colors.red

        print("CPU load - " + (cpu_color | f"{load_ratio:.2f}") + " | ", end='')

        try:
          r_smi = remote["nvidia_smi"]
        except plumbum.commands.processes.CommandNotFound: 
          print("")
          continue
        r_ps = remote["ps"]
        fieldnames = ['index', 'gpu_uuid', 'memory.total', 'memory.used',
                      'utilization.gpu', 'gpu_name']
        output = r_smi("--query-gpu=" + ",".join(fieldnames),
                       "--format=csv,noheader,nounits").replace(" ", "")

        gpu_data = []
        for line in output.splitlines():
            gpu_data.append(dict([(name, int(x)) if x.strip().isdigit() else (name, x)
                            for x, name in zip(line.split(","), fieldnames)]))

        for data in gpu_data:
            data['nr_processes'] = 0
            data['users'] = []
        ps_success = True
        if True:
          output = r_smi("--query-compute-apps=pid,gpu_uuid",
                         "--format=csv,noheader,nounits").replace(" ", "")

          gpu_processes = []
          for line in output.splitlines():
              gpu_processes.append([int(x) if x.strip().isdigit() else x for x in line.split(",")])

          ps_output, ps_success = timeout_call(r_ps['aux'])
          if ps_success:
            ps_users, ps_ids = [], []
            for line in ps_output.splitlines()[1:]:
              split = line.split()
              ps_users.append(split[0])
              ps_ids.append(int(split[1]))

            for process in gpu_processes:
                pid = process[0]
                if pid in ps_ids:
                  proc_idx = ps_ids.index(pid)
                  user = ps_users[proc_idx]
                else:
                  user = "unknown"
                serial = process[1]
                for data in gpu_data:
                    if data['gpu_uuid'] == serial:
                        data['users'].append(user.strip())
                        data['nr_processes'] += 1

        gpu_numbers = []
        gpu_status = []
        free_gpus = []

        all_users = set()
        for data in gpu_data:
            status = "    "+str(data['index']) + ": "
            # availability conditions: < 50MB and <5% utilisation ?

            # Is it free?
            if (data['memory.used'] < memory_threshold and
                data['utilization.gpu'] < gpu_util_threshold):

                status += "free"
                gpu_numbers.append(color_free | str(data['index']))
                free_gpus.append({'server': server,
                'gpu_nr': data['index'],
                                  'double': False})
                                  # 'session': getSession(data['index'])})
            else:
                all_users |= set(data['users'])
                status += "in use - " + str(data['users'])
                color = color_other
                gpu_numbers.append(color | str(data['index']))
                # print(data['users'])
                if (allow_lightly_used_gpus and
                    data['memory.used'] < upper_memory_threshold and
                    data['utilization.gpu'] < upper_gpu_util_threshold and
                    data['nr_processes'] < 4 and
                    # data['users'][0] in share_with):
                    (share_with == [] or not any([u not in share_with for u in data['users']]))):


                    free_gpus.append({'server': server,
                                      'gpu_nr': data['index'],
                                      'double': True})
                                      # 'session': getSession(data['index'] + 10)})
                    gpu_numbers[-1] = color_other_light | str(data['index'])
                    status = color_other_light | status
                elif (own_username in data['users']):
                    gpu_numbers[-1] = color_me | str(data['index'])
                    status = color_me | status
                else:
                    status = color | status

            gpu_status.append(status)
        print("GPUs - " + " ".join(gpu_numbers) + " | {} available".format(len(free_gpus)), end='')
        print(" | Users: " + ", ".join(sorted(all_users)))
        if verbose:
            print(f"  CPU ({int(num_cpus)} cores):")
            for u in sorted(cpu_usage, key=cpu_usage.get, reverse=True):
              usage = cpu_usage[u]
              if usage > 0.01:
                print(f"    {u:9} - {usage:.2f}")
            gpu_model = "{} - {} GB".format(gpu_data[0]['gpu_name'], gpu_data[0]['memory.total'] // 1000)
            print(f"  GPU ({gpu_model}):")
            if not ps_success:
              print("    NB: ps error, some processes not ID'd to user")
            for s in gpu_status:
                print(s)

        all_free_gpus += free_gpus
        remote.close()
    return all_free_gpus


def startExperimentSet(experiments,
                       free_gpus,
                       project_dir,
                       project_name,
                       repo_ssh_string,
                       sleep_time=10,
                       overrule_repo_check=False,
                       rebuild_docker=False,
                       repo_path="../"
                       ):
    """
    Requirements:
        ~/<project_dir>/docker/run.sh must exist and take two arguments
             1. The GPU to be used.
             2. The name of the docker contain to be created
             3. The command to be executed
        ~/<project_dir>/docker/build.sh must exist
        (That means that the docker script provided in pool/documentation needs to be adapted slightly)

        Also, the project directory on the server should be located in the home directory.

    Args
        experiments: List or dictionaries. Each list item corresponds to one experiment to be started.
                     The key-value pairs are the command line configs added to the call. (see example below)
        project_dir: Name of the project directory (assumed to be located in home directory).
        project_name: Name of the project. Is used to create the container name.
        repo_ssh_string: Github ssh string for repo. Used to clone or pull latest updates.
        sleep_time (int): Time in seconds to wait after starting the last experiment before checking whether all are running.
        overrule_repo_check: If true, don't check whether current repo is dirty.
        rebuild_docker: If true, rebuild docker container before starting experiments.
        repo_path: Path to local git repository.

    Return
        running_experiments: List of started experiments
        ids: List of sacred ids of started experiments
    """

    print("Starting experiments...")
    repo = Repo(repo_path)
    if overrule_repo_check:
        print("WARNING: No git check is performed!")
    else:
        assert not repo.is_dirty()

    running_experiments = []
    for gpu, exp in zip(free_gpus[:len(experiments)], experiments):
        print(exp)
        command = createCommand(exp, gpu)
        print(gpu, command)
        container_name = getSession(project_name, gpu)
        startExperiment(gpu, container_name, command, project_dir, repo_ssh_string, rebuild_docker=rebuild_docker)
        running_experiments.append({'gpu': gpu, 'config':exp})

    print('Waiting {} seconds...'.format(sleep_time))
    time.sleep(sleep_time)

    print('Checking whether experiments are running...')
    ids = []
    for started_experiment in running_experiments:
        id_ = inspect_container(project_name, started_experiment['gpu'])
        print(id_, started_experiment)
        ids.append(id_)

    return running_experiments, ids


###################
# Private Methods #
###################


def createCommand(exp, gpu):
    command = "./code/main.py -p with " + " ".join(["{}={}".format(key, exp[key]) for key in exp])
    command += " server.name={} server.gpu_id={}".format(gpu['server'], gpu['gpu_nr'])
    return command


def inspect_container(project_name, gpu):
    remote = SshMachine(gpu['server'])
    session = getSession(project_name, gpu)

    r_docker = remote['docker']
    inspect = r_docker('inspect', session)
    data = json.loads(inspect)
    # print(data[0]['State']['Running'])
    if data[0]['State']['Running']:
        logs = r_docker('logs', session)
        m = re.search('Started run with ID "(\d+)"', logs)
        id_ = int(m.group(1))
        return id_
    else:
        raise Exception("Container not running")


def startExperiment(gpu, session, command, project_dir, repo_ssh_string, rebuild_docker=False):
    """
    Helper function to start an experiment remotely.

    Requirements:
        <project_dir>/docker/run.sh must exist and take two arguments
             1. The name of the docker contain to be created
             2. The command to be executed
        <project_dir>/docker/build.sh must exist

        Also, the project directory should be located in the home directory.

    Args:
        gpu (int): Id of the GPU
        session (string): Name of the container to be created
        command (string): Command to be exectued
        project_dir (string): Name of the project directory

    """
    remote = SshMachine(gpu['server'])
    r_runDocker = remote[remote.cwd / project_dir / "docker/run.sh"]
    r_buildDocker = remote[remote.cwd / project_dir / "docker/build.sh"]
    home_dir = remote.cwd

    killRunningSession(remote, session)
    updateRepo(remote, home_dir, project_dir, repo_ssh_string)

    if rebuild_docker:
        # Build docker
        print("Building container...")
        with remote.cwd(home_dir / project_dir / 'docker/'):
            r_buildDocker()
            print("Done.")

    with remote.cwd(home_dir / project_dir):
        # r_runDocker(gpu, "code/main.py -p with ./code/conf/openaiEnv.yaml")
        print("Running command: {}".format(command))
        r_runDocker(str(gpu['gpu_nr']), session, command)

    remote.close()


def updateRepo(remote, home_dir, project_dir, repo_ssh_string):
    """
    Helper function to pull newest commits to remote repo.
    """
    r_git = remote['git']
    # Update repository
    try:
        print("Cloning repository...")
        with remote.cwd(home_dir):
            r_git("clone", repo_ssh_string)
    except plumbum.commands.ProcessExecutionError as e:
        if e.stderr.find('already exists') == -1:
            raise e
        else:
            print("Git directory already exists. Fetching new commits...")
        with remote.cwd(home_dir / project_dir):
            r_git('fetch', '-q', 'origin')
            r_git('reset', '--hard', 'origin/master', '-q')

    # Check that we have the same git hash remote than local
    with remote.cwd(home_dir / project_dir):
        r_head = r_git('rev-parse', 'HEAD')
    l_head = git('rev-parse', 'HEAD')
    assert l_head == r_head, "Local git hash != pushed git hash. Did you forget to push changes?"


def killRunningSession(remote, session):
    """
    Help function to kill running containers to start new ones with the same name.
    """
    r_docker = remote['docker']
    try:
        r_docker("stop", session)
        r_docker("rm", session)
        print("Killed running session {}".format(session))
    except plumbum.commands.ProcessExecutionError as e:
        print(e)
        pass


def getSession(project_name, gpu):
    """
    Create a name for the container to be started.
    Returns "<project_name><gpu_nr+1>".
    The +1 is to start counting at 1 because 0 creates weird problems.
    """
    nr = gpu['gpu_nr'] + 1
    if gpu['double']:
        nr += 10
    session = "{}{}".format(project_name, nr)
    return session

def inspect_gpu_wrapper(input):
  server, own_username = input
  try:
    return inspect_gpus(servers=[server], verbose=False, own_username=own_username)
  except Exception as e: 
    print(e)
    return None


if __name__ == "__main__":
    args = parser.parse_args()

    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(dt_string)

    print("GPU color codes: " +
          (color_free | "Free" + " | ") +
          (color_me | "Own" + " | ") +
          (color_other | "Other" + " | ") +
          (color_other_light | "Other (light)"))
    # flair_servers = ['f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9']
    flair_servers = ['flair12']
    for server in args.ignore:
        if server in flair_servers:
            flair_servers.remove(server)
    # hercules timeout
    # dgx1 login not working
    # mulga seomtimes broken and brown removed since broken

    from multiprocessing import Pool
    print('-'*8 + 'Flair' + '-'*8)
    with Pool(processes=4) as p_pool: # issues with too many processes
      p_pool.map(inspect_gpu_wrapper, zip(flair_servers, ['michaelm']*len(flair_servers)))
