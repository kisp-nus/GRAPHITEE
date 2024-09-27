import argparse
import dgl


BASE_ETYPES = ['history', 'ne_link', 'nw_link', 'ue_link', 'uw_link']


def load_graph(mind_dgl):
    support_edges = {
        ('user', 'history', 'news'): list(range(mind_dgl.num_relation['history'])), 
        # ('news', 'ne_link', 'entity'): list(range(mind_dgl.num_relation['ne_link'])), 
        # ('news', 'nw_link', 'word'): list(range(mind_dgl.num_relation['nw_link'])), 
        # ('user', 'ue_link', 'entity'): list(range(mind_dgl.num_relation['ue_link'])), 
        # ('user', 'uw_link', 'word'): list(range(mind_dgl.num_relation['uw_link'])), 
        ('news', 'history_r', 'user'): list(range(mind_dgl.num_relation['history'])), 
        # ('entity', 'ne_link_r', 'news'): list(range(mind_dgl.num_relation['ne_link'])), 
        # ('word', 'nw_link_r', 'news'): list(range(mind_dgl.num_relation['nw_link'])), 
        # ('entity', 'ue_link_r', 'user'): list(range(mind_dgl.num_relation['ue_link'])), 
        # ('word', 'uw_link_r', 'user'): list(range(mind_dgl.num_relation['uw_link'])), 
        ('user', 'user_selfloop', 'user'): list(range(mind_dgl.num_relation['user_selfloop'])), 
        ('news', 'news_selfloop', 'news'): list(range(mind_dgl.num_relation['news_selfloop'])), 
        # ('entity', 'entity_selfloop', 'entity'): list(range(mind_dgl.num_relation['entity_selfloop'])), 
        # ('word', 'word_selfloop', 'word'): list(range(mind_dgl.num_relation['word_selfloop'])), 
    }
    sub_g = dgl.edge_subgraph(mind_dgl.graph, support_edges)
    return dgl.to_homogeneous(sub_g)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Node2vec')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=5)
    parser.add_argument('--p', type=float, default=0.25)
    parser.add_argument('--q', type=float, default=4.0)
    parser.add_argument('--num_walks', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    return args