from dgl.data import FlickrDataset
from generate_dataset import generate_dataset

def main():
    dataset = FlickrDataset()
    flickr_graph = dataset[0]
    generate_dataset('flickr', flickr_graph, 'feat', 180, 32, [0.85, 0.05, 0.1])

if __name__ == '__main__':
    main()