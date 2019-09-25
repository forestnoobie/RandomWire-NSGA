import networkx as nx
import collections

# collections.namedtuple(typename, field_names)
# typename의 이름으로 class가 정의되며, field_names로 접근 가능
# id에 해당하는 node에 들어오는 노드들의 id가 inputs에 리스트로 저장됨
# ex) id = 10, inputs = [1, 3, 4, 6]

Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])

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
    Nodes.append(Node(node, [n for n in tmp if n < node], type))    # DAG(Directed Acyclic Graph)로 변환
  return Nodes, input_nodes, output_nodes

def build_graph(Nodes, args):
  if args.graph_model == 'ER':
    return nx.random_graphs.erdos_renyi_graph(Nodes, args.P, args.seed)
  elif args.graph_model == 'BA':
    return nx.random_graphs.barabasi_albert_graph(Nodes, args.M, args.seed)
  elif args.graph_model == 'WS':
    return nx.random_graphs.connected_watts_strogatz_graph(Nodes, args.K, args.P, tries=200, seed=args.seed)

def save_graph(graph, path):
  nx.write_yaml(graph, path)

def load_graph(path):
  return nx.read_yaml(path)