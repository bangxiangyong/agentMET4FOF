import os
import datetime
import pickle

def get_pipeline_details(pipelines):
    pipeline_details = [pipeline.agents(ret_hyperparams=True) for pipeline in pipelines]
    return pipeline_details

def save_experiment(ml_experiment):
    name = ml_experiment.name
    file = open(ml_experiment.path+"/"+name+".pkl", 'wb')
    pickle.dump(ml_experiment, file)
    file.close()

def load_experiment(ml_experiment_name="run_1",experiment_folder="ML_EXP"):
    file_path = experiment_folder+"/"+ml_experiment_name+"/"+ml_experiment_name+".pkl"
    if os.path.exists(file_path):
        try:
            file = open(file_path, 'rb')
            # dump information to that file
            data = pickle.load(file)
            file.close()
            return data
        except Exception as e:
            print("Error in loading experiment: "+ str(e))
            return -1
    else:
        return -1


class ML_Experiment:
    def __init__(self, agentNetwork, datasets=[], pipelines=[], evaluation=[], name="run", train_mode={"Prequential","Kfold5","Kfold10"}):

        experiment_folder = "ML_EXP"
        if type(pipelines) is not list:
            pipelines = [pipelines]
        if type(datasets) is not list:
            datasets = [datasets]
        if type(evaluation) is not list:
            evaluation = [evaluation]

        #create new directory
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)

        #create new directory for name with running number to ensure unique runs
        running_number = 1
        name= name.lower()
        directory_lists_filter = [dir_.lower() for dir_ in os.listdir(experiment_folder) if name in dir_]
        if len(directory_lists_filter) == 0:
            name = name+"_1"
            os.makedirs(experiment_folder+"/"+name)
        else:
            directory_lists_filter = [int(dir_.split("_")[-1]) for dir_ in os.listdir(experiment_folder) if name in dir_]
            directory_lists_filter.sort()
            next_count = int(directory_lists_filter[-1])+1
            name = name+"_" +str(next_count)
            os.makedirs(experiment_folder+"/"+name)

        #record all the meta data
        self.name = name
        self.path = experiment_folder+"/"+self.name
        self.run_date = datetime.datetime.today()
        self.run_date_string = self.run_date.strftime("%d-%m-%y %H:%M:%S")
        self.pipeline_details = get_pipeline_details(pipelines)
        self.chain_results = []
        self.datasets = []

        #handle collecting dataset names
        #handle binding of agents and pipelines
        for datastream_agent in datasets:
            self.datasets.append(datastream_agent.get_attr('data_name'))
            for pipeline in pipelines:
                for evaluation_agent in evaluation:
                    pipeline.bind_output(evaluation_agent)

                    datastream_agent.bind_output([agent_proxy for agent_proxy in pipeline.pipeline[0]])

        #handle training mode
        if type(train_mode) == set:
            self.train_mode = "Kfold5"
        else:
            self.train_mode = train_mode

        #connect agent network
        if agentNetwork is not None:
            self.setup_ml_logger(agentNetwork,self)

    def update_chain_results(self,res):
        self.chain_results.append(res)


    def setup_ml_logger(self, agentNetwork, ml_experiment):
        def set_ml_experiment(self, ml_experiment=False):
            self.ml_experiment = ml_experiment

        def log_handler_ML(self, message, topic):
            """
            Handles the results coming from Evaluation agent to be saved into the provided ML experiment file.
            This updates the results of individual "chains" to be aggregated later for comparisons of chains/pipelines
            The mechanism relies on regularly saving the ml_experiment object into the pickled file in default ML_EXP folder.

            """
            if self.ml_experiment:
                self.ml_experiment.update_chain_results(message)
                save_experiment(self.ml_experiment)

        #update logger agent
        logger_agent = agentNetwork._get_logger()
        logger_agent.set_method(set_ml_experiment=set_ml_experiment)
        logger_agent.set_ml_experiment(ml_experiment)
        logger_agent.bind_log_handler({"ML_EXP":log_handler_ML})
