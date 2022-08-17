#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:18:22 2021

@author: skibbe
"""
import numpy as np
from os import listdir
import os
from os.path import isfile, join, isdir
import json
import time
import subprocess


def bash_run(command):

        proc = subprocess.Popen(['/bin/bash'],text=True ,stdin=subprocess.PIPE, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return proc.communicate(command) , (proc.returncode == 0)

def slurm_submit(commands,ofile,**kwargs):
        params={};
        params['quiet']=False;
        params['debug']=False;
        #params['debug']=True;
        params['mem']=5000;
        params['cores']=2;
        params['append']=False;
        params['name']='kakushin pipeline';
        params['time']='01-00:00:00';
        params['queue']='kn_pipe_master';
        params['feature']='';
        params['gres']='';
        params['nodelist']='';
        params['exclude']='';


        if kwargs is not None:
            for key, value in kwargs.items():
            #print "%s == %s" %(key,value)
                params[key]=value;

        #if 'commands' not in params:
        #       print 'ERROR: no commands given'
        #       return


        if not (os.access(os.path.dirname(ofile), os.W_OK)):
            print("#E: cannot write to the folder "+os.path.dirname(ofile))
            print("#E: to store the file "+ofile)
            return "0", False;

        #params['queue']='gpucpu';
        #params['time']='01-00:00:00';

        params['name']=params['name'].replace(' ','-');

        batch='printf "';
        batch=batch+'#!/bin/bash \\n';

        batch=batch+'#SBATCH --job-name=\"'+format(params['name'])+'\"\\n';
        #batch=batch+'#SBATCH --job-name=\"sds\"\\n';
        batch=batch+'#SBATCH -c '+format(params['cores'])+'\\n';
        batch=batch+'#SBATCH --mem '+format(params['mem'])+'\\n';
        batch=batch+'#SBATCH -t '+format(params['time'])+'\\n';
        batch=batch+'#SBATCH --error '+ofile+'\\n';
        batch=batch+'#SBATCH --output '+ofile+'\\n';
        batch=batch+'#SBATCH -p '+format(params['queue'])+'\\n';
        if len(params['feature'])>0:
            batch=batch+'#SBATCH --constraint=\"'+format(params['feature'])+'\"\\n';

        if len(params['nodelist'])>0:
            batch=batch+'#SBATCH --nodelist='+format(params['nodelist'])+'\\n';
        if len(params['exclude'])>0:
            batch=batch+'#SBATCH --exclude='+format(params['exclude'])+'\\n';


        if len(params['gres'])>0:
            batch=batch+'#SBATCH --gres=\"'+format(params['gres'])+'\"\\n';

        #batch=batch+'echo job id is :${SLURM_JOBID}\\n';

        if params['append']:
            batch=batch+'#SBATCH --open-mode append\\n';
        else:
            batch=batch+'#SBATCH --open-mode truncate\\n';

        for command in commands:
            #command = command.replace('"','\"'); #should be added, but for compatibility is not (yet)
            command = command.replace('$','\$');
            #print command
            batch=batch+command+'\\n';

        #batch=batch+"sacct -o reqmem,maxrss,averss,elapsed -j \$SLURM_JOBID\\n";
        #batch=batch+"sacct -o reqmem,maxrss,averss,AllocCPUs,AveCPU,AveCPUFreq,MaxDiskWrite,MaxDiskRead,AveDiskRead,elapsed -j \$SLURM_JOBID\\n";
        #sacct -p -o  JobID,reqmem,maxrss,averss,AllocCPUs,AveCPU,AveCPUFreq,MaxDiskWrite,AveDiskWrite,MaxDiskRead,AveDiskRead,elapsed -j 32997

        batch=batch+'" | sbatch';
        #batch=batch+'" ';

        if      params['debug']:
            batch="";
            for command in commands:
                command.replace('"','\"');
                batch=batch+command+';';
                res, success=bash_run(batch);
                res="0";

        else:

            if not params['quiet']:
                print("#I: BATCH SCRIPT: ---------------------------------")
                print(batch)
                print("#I: -----------------------------------------------")
            batch='histchars=;'+batch+';unset histchars;'
            res, success=bash_run(batch);

            #print(format(res))
            if success:
                #print res
                res=res[0].split()[-1];
                if not params['quiet']:
                    print(res)
                job_state, success=bash_run('squeue -h  --job '+res+'  -o "%t"')
                if success:
                    job_state=[f for f in job_state[0].split("\n") if (len(f)>0)];
                    if len(job_state)==1:
                        success = ( job_state[0] in {'R','PD','CF','CG'} )
                    else:
                        print("#E: cannot find the JOB in the queue")
                        success=False

        return res, success;
    
def wait_for_jobs(job_ids):
    if len(job_ids)==0:
        return False, True, 1

    jobids=",".join([str(f) for f in job_ids])

    job_state, success=bash_run('squeue -h  --job '+jobids+'  -o "%t"')

    is_running=False;

    if success:
        job_state=[f for f in job_state[0].split("\n") if (len(f)>0)];
        progress=len(job_ids)-len(job_state);
        for job in job_state:
            running = ( job  in {'R','PD','CF','CG'} )
            if not running:
                progress += 1;
            if ( job in {'F','CA','TO','NF','SW'} ):
                return False,False, 0
            is_running=(is_running or running)

    else:
        progress = len(job_ids);

    return is_running, True, (float(progress)/float(len(job_ids)))


def get_jobid_string(jobids):
    job_str="";
    count=1;
    prev_id=-1;
    for f in jobids:
        current_id=int(f);
        if count==1:
            job_str+='{'+format(current_id)+'.'
            prev_id=current_id;

        if (current_id-prev_id)>1:
            job_str+='.'+format(prev_id)+'} '+'{'+format(current_id)+'.'
        if count==len(jobids):
            job_str+='.'+format(current_id)+'}'

        prev_id=current_id;
        count=count+1;
    return job_str


def my_wait_for_jobs(jobids,progress_scale=(0,1)):
        progress_old=-1;
        is_running=True;
        while (is_running):
            is_running, success, progress =  wait_for_jobs(jobids)
            if not success:
                jobids=",".join([str(f) for f in jobids])
                print("#W: warning, killing jobs")
                job_state, success=pipetools.bash_run('scancel '+jobids)
                if not success:
                    print("#W: "+job_state[1])
            else:
                if progress!=progress_old:
                    progress_old=progress;
                    print(int(100*(progress_scale[0]+progress_scale[1]*progress)))
            if is_running:
                time.sleep( 10 )

        if not success:
            raise CPipeError("tracking jobs failed");

print("ok")