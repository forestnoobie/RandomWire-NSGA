import networkx as nx
import collections

import glob
from easydict import EasyDict
import random

# collections.namedtuple(typename, field_names)
# typename의 이름으로 class가 정의되며, field_names로 접근 가능
# id에 해당하는 node에 들어오는 노드들의 id가 inputs에 리스트로 저장됨
# ex) id = 10, inputs = [1, 3, 4, 6]

Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])  # typename, field_names


def build_graph(Nodes, args):
    if args.graph_model == 'ER':
        return nx.random_graphs.erdos_renyi_graph(Nodes, args.P)
    elif args.graph_model == 'BA':
        return nx.random_graphs.barabasi_albert_graph(Nodes, args.M)
    elif args.graph_model == 'WS':
        return nx.random_graphs.connected_watts_strogatz_graph(Nodes, args.K, args.P, tries=200)


def save_graph(graph, path):
    nx.write_yaml(graph, path)
    
    
def load_graph(path):
    return nx.read_yaml(path)
    

def get_graph_info(graph):
    input_nodes = []
    output_nodes = []
    Nodes = []
    for node in range(graph.number_of_nodes()):
        # node i 에 대해        
        tmp = list(graph.neighbors(node))
        tmp.sort()    # 오름차순 정렬
    
        # node type 정의    
        type = -1    # input node도, output node도 아닌. 그래프의 중간에 매개자처럼 있는 중간 node.
        if node < tmp[0]:
            input_nodes.append(node)
            type = 0    # id 가장 작은 노드보다 작으면, 이건 외부에서 input을 받는 노드. 즉 input node.
        if node > tmp[-1]:
            output_nodes.append(node)
            type = 1    # id 가장 큰 노드보다 크면, 이건 외부로 output 내보내는 노드. 즉 output node.
        
        # dag로 변환 (자신의 id보다 작은 노드들과의 연결만 남기기)
        # [type] 0: input node, 1: output node, -1: input도 output도 아닌, 그래프 중간에 매개자처럼 있는 중간 node
        Nodes.append(Node(node, [n for n in tmp if n < node], type))    # DAG(Directed Acyclic Graph)로 변환
    return Nodes, input_nodes, output_nodes


# random graph를 'num_graph' 개수만큼 만든 후, stage_pool_path에 .yaml 파일로 저장해주는 함수
#     - Nodes 수, K, P의 범위는 임의로 정함.
#     - 일단 WS로만 만듦
def make_random_graph(num_graph, stage_pool_path):
    print("Start to make random graph pool...")
    check_path = glob.glob(stage_pool_path + '*.yaml')
    check_file_num = len(check_path)

    while check_file_num < num_graph:

        Nodes = random.randint(20, 40)  # => [Nodes 값 수정 시 주의] 아래 K값은 Nodes보다 작아야함.
        graph_model = 'WS'
        K = random.randint(4, Nodes-10)  # [min, max] // WS에서는 K nearest. 따라서, 4 ~ 30 random 선택 하도록
        P = round(random.uniform(0.25, 0.75), 2)   # 소수 둘째자리까지만 나오도록 반올림 => 결과 e.g. 0.75

        args = EasyDict({'graph_model': graph_model, 'K':K, 'P':P})

        P_str = str(P)[0] + str(P)[2:]   # 0.75 => 075
        save_file_path = stage_pool_path + graph_model + '_' + str(Nodes) + '_' + str(K) + '_' + P_str + '.yaml'   # e.g. WS_32_4_075

        graph = build_graph(Nodes, args)
        save_graph(graph, save_file_path)

        check_path = glob.glob(stage_pool_path + '*.yaml')
        check_file_num = len(check_path)
    print("Finished")