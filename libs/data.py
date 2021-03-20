#!python

# Necessary modules
import os
import requests
import sys
import scipy.io
import shutil


class StanfordDataset:

    def __init__(self):

        self.metadata_url = 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
        self.test_meta_url = 'http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat'
        self.train_dataset_url = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
        self.test_dataset_url = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'

        self.train_filename = 'train.tgz'
        self.test_filename = 'test.tgz'
        self.metadata_file_name = 'metadata.tgz'
        self.test_meta_filename = 'test_with_labels.mat'

        # save location where the data will be stored
        self.raw_loc = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw'))
        # save location for clean data
        self.clean_data_loc = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/clean'))

        self.train_raw_path = os.path.join(self.raw_loc, 'cars_train')
        self.test_raw_path = os.path.join(self.raw_loc, 'cars_test')

        self.labels = list()
        self.train_data = tuple()
        self.test_data = tuple()

        # clean data paths
        self.train_clean_path = os.path.join(self.clean_data_loc, 'train')
        self.test_clean_path = os.path.join(self.clean_data_loc, 'test')

    def download(self, url, file_name):

        file_abs_loc = os.path.join(self.raw_loc, file_name)

        with open(file_abs_loc, 'wb') as file_obj:
            response = requests.get(url, stream=True)

            _good_response = response.status_code != 404

            if _good_response:
                file_size = int(response.headers.get('Content-Length'))
                complete = 0
                complete_perc = 0
                file_size_mb = self._convert_bytes_to_mega_bytes(file_size)
                complete_mb = 0

                print('Downloading {} from {}'.format(file_name, url))
                for data in response.iter_content(chunk_size=2048):
                    file_obj.write(data)
                    string = '[{}{}] {}/{} MB Complete'.format('|' * (complete_perc + 1),
                                                               ' ' * (99 - complete_perc),
                                                               complete_mb,
                                                               file_size_mb
                                                               )

                    sys.stdout.write('\r' + string)
                    sys.stdout.flush()

                    complete += len(data)
                    complete_perc = int((complete/file_size) * 100)
                    complete_mb = self._convert_bytes_to_mega_bytes(complete)

                print('\n')

    def download_test_meta(self):
        self.download(self.test_meta_url, self.test_meta_filename)

    def download_train_dataset(self):
        self.download(self.train_dataset_url, self.train_filename)

    def download_test_dataset(self):
        self.download(self.test_dataset_url, self.test_filename)

    def download_meta(self):
        self.download(self.metadata_url, self.metadata_file_name)

    def fetch_data(self):
        self.download_meta()
        self.download_test_meta()
        self.download_train_dataset()
        self.download_test_dataset()

    def load_data(self, download=False):

        # If this flag is set then the data is downloaded
        # from scratch.
        if download:
            self.fetch_data()
            self.extract()

        self.get_labels()
        self.populate_data()

    def extract(self):

        train_set = os.path.join(self.raw_loc, self.train_filename)
        test_set = os.path.join(self.raw_loc, self.test_filename)
        metadata = os.path.join(self.raw_loc, self.metadata_file_name)

        print('Extracting Training dataset...')
        try:
            if os.path.isfile(train_set):
                shutil.unpack_archive(train_set, self.raw_loc)
        except Exception as ex:
            print("Warning!!! Can not extract training dataset")

        print('Extracting Testing dataset...')
        try:
            if os.path.isfile(test_set):
                shutil.unpack_archive(test_set, self.raw_loc)
        except Exception as ex:
            print("Warning!!! Can not extract testing dataset")

        print('Extracting Metadata...')
        try:
            if os.path.isfile(metadata):
                shutil.unpack_archive(metadata, self.raw_loc)
                # After metadata is extracted use the info to set labels
                self.get_labels()

        except Exception as ex:
            print("Warning!!! Can not extract Metadata")

    def populate_data(self):
        # clean train datasets
        annos_train_metadata_pth = os.path.join(self.raw_loc, 'devkit/cars_train_annos.mat')
        # clean test datasets
        annos_test_metadata_pth = os.path.join(self.raw_loc, 'test_with_labels.mat')

        # loading train annotation file
        annos_train = scipy.io.loadmat(annos_train_metadata_pth)
        annos_train_data = annos_train.get('annotations')[0]

        # loading test annotation file
        annos_test = scipy.io.loadmat(annos_test_metadata_pth)
        annos_test_data = annos_test.get('annotations')[0]

        # get class labels and file_name for train data
        train_data = tuple(map(lambda x: (self.labels[x[4][0][0] - 1], x[5][0]), annos_train_data))

        # get class labels and file_name for test data
        test_data = tuple(map(lambda x: (self.labels[x[4][0][0] - 1], x[5][0]), annos_test_data))

        # setting the global values
        self.train_data = train_data
        self.test_data = test_data

    @staticmethod
    def _convert_bytes_to_mega_bytes(bdata, roundit=2):
        byte_to_mb_conv = 1048576
        return round(bdata / byte_to_mb_conv, roundit)

    def get_labels(self):
        # mat file location that contains class label information
        mat_file_abs_path = os.path.join(self.raw_loc, 'devkit/cars_meta.mat')
        # Loading mat file data
        mat_data = scipy.io.loadmat(mat_file_abs_path)

        # extracting class labels
        class_labels = list(map(lambda x: x[0], mat_data.get('class_names')[0]))

        # setting labels attribute
        self.labels = class_labels


if __name__ == '__main__':
    obj = StanfordDataset()

    # obj.download(obj.metadata_url, obj.metadata_file_name)
    # obj.fetch_data()

    # obj.fetch_data()
    # obj.fetch_data()

    # obj.extract()

    obj.get_labels()

    # print(obj.labels)
    obj.load_data()
    print(obj.test_data)
    print(obj.train_data)