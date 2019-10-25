import os
import sys
import os
import logging
from easydict import EasyDict
import numpy as np
import random
import time
import datetime
from deap import tools
from collections import OrderedDict
from pprint import pprint
import json
import torch

sys.path.insert(0, '../')
from utils_kyy.utils_graph import make_random_graph
from utils_kyy.create_toolbox import create_toolbox_for_NSGA_RWNN
from utils_kyy.create_toolbox import evaluate


class rwns_train:
    def __init__(self, json_file):
        self.root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        # self.root = os.getcwd()
        self.param_dir = os.path.join(self.root + '/parameters/', json_file)
        f = open(self.param_dir)
        params = json.load(f)
        pprint(params)
        self.name = params['NAME']
        self.log_dir = os.path.join(self.root, 'log')
        self.model_dir = os.path.join(self.root + '/saved_models/', self.name)

        # create directory
        if not (os.path.isdir(self.model_dir)):
            os.makedirs(self.model_dir)
        if not (os.path.isdir(self.log_dir)):
            os.makedirs(self.log_dir)

        ## toolbox params
        self.args_train = EasyDict(params['ARGS_TRAIN'])
        self.data_path = params['DATA_PATH']
        self.run_code = params['RUN_CODE']
        self.stage_pool_path = '../graph_pool' + '/' + self.run_code + '/'
        self.log_path = '../logs/' + self.run_code + '/'
        self.num_graph = params['NUM_GRAPH']
        self.toolbox = None

        ## GA params
        self.pop_size = params['POP_SIZE']
        self.ngen = params['NGEN']
        self.cxpb = params['CXPB']
        self.mutpb = params['MUTPB']

        ## logs
        log = OrderedDict()

        log['hp'] = self.args_train
        self.log = log
        self.train_log = None

        ## tool box and make graph

    def create_toolbox(self):
        self.stage_pool_path = '../graph_pool' + '/' + self.run_code + '/'
        self.log_path = '../logs/' + self.run_code + '_' + self.name + '/'

        if not os.path.exists(self.stage_pool_path): os.makedirs(self.stage_pool_path)
        if not os.path.isdir(self.log_path): os.makedirs(self.log_path)
        self.log_file_name = self.log_path + 'logging.log'
        self.train_log_file_name = self.log_path + 'train_logging.log'

        logging.basicConfig(filename=self.log_file_name, level=logging.INFO)
        logging.info('Start to write log.')

        # num_graph = 100
        make_random_graph(self.num_graph, self.stage_pool_path)

        return create_toolbox_for_NSGA_RWNN(self.num_graph, self.args_train, self.stage_pool_path, self.data_path,
                                            self.log_file_name)

    ## Train

    def train(self):

        ## Parameters
        self.create_toolbox()
        POP_SIZE = self.pop_size

        ## train log
        train_log = OrderedDict()

        if self.toolbox is None:
            self.toolbox = self.create_toolbox()

        toolbox = self.toolbox

        # log에 기록할 stats
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "min", "max", "evals_time", "gen_time"

        # population 생성.  (toolbox.population은 creator.Individual n개를 담은 list를 반환. (=> population)
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        print("Initialion starts ...")
        logging.info("Initialion starts at " + now_str)
        init_start_time = time.time()

        pop = toolbox.population(n=POP_SIZE)

        ## fitness, model list
        fit_list = []
        model_list = []

        local_min_fit1 = float('inf')
        local_min_fit2 = float('inf')
        local_min_index = [None, None]

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # .evaluate는 tuple을 반환. 따라서 fitnesses는 튜플을 원소로 가지는 list

        for idx, ind in enumerate(invalid_ind):
            fitness, ind_model = evaluate(ind, args_train=self.args_train, stage_pool_path=self.stage_pool_path,
                                          data_path=self.data_path, log_file_name=self.log_file_name)
            ind.fitness.values = fitness
            fit_list.append(fitness)
            model_list.append(ind_model)

            if fitness[0] < local_min_fit1:
                local_min_fit1 = fitness[0]
                local_min_index[0] = idx

            if fitness[1] < local_min_fit2:
                local_min_fit2 = fitness[1]
                local_min_index[1] = idx

        ## index ckpt download
        print("#### Saving Model", local_min_index)
        self.save_model(model=model_list[local_min_index[0]], ngen=0, subname=str(0) + '_' + 'acc')
        self.save_model(model=model_list[local_min_index[1]], ngen=0, subname=str(0) + '_' + 'flops')

        ## log 기록
        train_log[0] = fit_list
        self.train_log = train_log

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)

        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        print("Initialization is finished at", now_str)
        logging.info("Initialion is finished at " + now_str)

        init_time = time.time() - init_start_time
        logging.info("Initialion time = " + str(init_time) + "s")

        print()

        # Begin the generational process
        for gen in range(1, self.ngen):
            now = datetime.datetime.now()
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')
            print("#####", gen, "th generation starts at", now_str)
            logging.info(str(gen) + "th generation starts at" + now_str)

            start_gen = time.time()
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.cxpb:
                    toolbox.mate(ind1, ind2)

                toolbox.mutate(ind1, indpb=self.mutpb)
                toolbox.mutate(ind2, indpb=self.mutpb)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            print("##### Evaluation starts")
            start_time = time.time()

            ## fitness value : accuracy ,flops 모음
            fit_list = []
            model_listist = []

            ## 가장 최소 1,2 구하기
            local_min_fit1 = float('inf')
            local_min_fit2 = float('inf')
            local_min_index = [None, None]

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            for idx, ind in enumerate(invalid_ind):
                fitness, ind_model = evaluate(ind, args_train=self.args_train, stage_pool_path=self.stage_pool_path,
                                              data_path=self.data_path, log_file_name=self.log_file_name)
                ind.fitness.values = fitness
                fit_list.append(fitness)
                model_list.append(ind_model)

                if fitness[0] < local_min_fit1:
                    local_min_fit1 = fitness[0]
                    local_min_index[0] = idx

                if fitness[1] < local_min_fit2:
                    local_min_fit2 = fitness[1]
                    local_min_index[1] = idx

            ## index ckpt download
            print("#### Saving Model", local_min_index)
            self.save_model(model=model_list[local_min_index[0]], ngen=gen, subname=str(gen) + '_' + 'acc')
            self.save_model(model=model_list[local_min_index[1]], ngen=gen, subname=str(gen) + '_' + 'flops')

            ## log 기록
            train_log[gen] = fit_list
            self.train_log = train_log

            eval_time_for_one_generation = time.time() - start_time
            print("##### Evaluation ends (Time : %.3f)" % eval_time_for_one_generation)

            # Select the next generation population
            pop = toolbox.select(pop + offspring, POP_SIZE)

            gen_time = time.time() - start_gen
            print('##### [gen_time: %.3fs]' % gen_time, gen, 'th generation is finished.')

            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record,
                           evals_time=eval_time_for_one_generation, gen_time=gen_time)

            logging.info('Gen [%03d/%03d] -- evals: %03d, evals_time: %.4fs, gen_time: %.4fs' % (
                gen, self.ngen, len(invalid_ind), eval_time_for_one_generation, gen_time))
            print(logbook.stream)

    ## Save Check point

    ## Save Log
    def save_log(self):

        log = self.log

        ## 필요한 log 추후 정리하여 추l가
        log['train_log'] = self.train_log

        with open(self.train_log_file_name, 'w', encoding='utf-8') as make_file:
            json.dump(log, make_file, ensure_ascii=False, indent='\t')

    ## Save Model
    def save_model(self, model, ngen, subname):

        model_fname = self.name + '_' + str(ngen) + '_' + subname
        model_path = os.path.join(self.model_dir, model_fname)
        print("Saving Model", model_path)
        torch.save(model, model_path)
