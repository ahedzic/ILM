from dgl.data import YelpDataset
from generate_dataset import generate_dataset

def main():
    dataset = YelpDataset()
    yelp_graph = dataset[0]
    generate_dataset('yelp', yelp_graph, 'feat', 200, 32, [0.85, 0.05, 0.1])

if __name__ == '__main__':
    main()
