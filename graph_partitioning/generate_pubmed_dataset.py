from dgl.data import PubmedGraphDataset
from generate_dataset import generate_dataset

def main():
    dataset = PubmedGraphDataset()
    pubmed_graph = dataset[0]
    generate_dataset('pubmed', pubmed_graph, 'feat', 200, 5, [0.85, 0.05, 0.1])

if __name__ == '__main__':
    main()
