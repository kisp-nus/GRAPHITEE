import warnings
warnings.filterwarnings('ignore')
import os, sys
home_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, home_dir)
# --------------------------------------------------------------------

from data.utils.node2vec.model import Node2vecModel
from data.utils.node2vec.utils import load_graph, parse_arguments


def train_node2vec(graph, args):
    """
    Train node2vec model
    """
    trainer = Node2vecModel(graph,
                            embedding_dim=args.embedding_dim,
                            walk_length=args.walk_length,
                            p=args.p,
                            q=args.q,
                            num_walks=args.num_walks,
                            device=args.device)
    
    trainer.train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=0.01)


if __name__ == '__main__':
    args = parse_arguments()
    graph = load_graph()
    print(graph)
    print("Perform training node2vec model")
    train_node2vec(graph, args)