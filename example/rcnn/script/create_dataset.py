import os
import pandas as pd
import zipfile
import sys
import tarfile
import cv2

input_path = '/home/ubuntu/workspace/mxnet/example/rcnn/new_data/VOCdevkit/VOC2007'
outdir = '/home/ubuntu/workspace/mxnet/example/rcnn/new_data/VOCdevkit/VOC2007/Raw/Out/'

def extract_folders(indir):
	for root, dirs, filenames in os.walk(indir):
		#print(root)
		#print(dirs)
		for f in filenames:
			#print(f)
			if f.endswith('.zip'):
				tar = tarfile.open(indir + str(f))
				tar.extractall(path=outdir)
				tar.close()


def rename_filenames(outdir):
	file_list = []

	for path, subdirs, files in os.walk(outdir):
		#print('Path: ' + str(path))
		#print('Subdirs: ' + str(subdirs))
		#print('Files: ' + str(files))
		tmp_file_list = []
		for name in files:
			#print(name)
			if name.endswith('.pdf'):
				pdf_name, file_ending = name.split('.pdf')
			elif name.endswith('.PDF'):
				pdf_name, file_ending = name.split('.PDF')
			if name.endswith('.json'):
				json_name = name
			if name.endswith('.jpeg'):
				tmp_file_list.append(name)
            	#file_name, file_ending = name.split('.jpeg')
		tmp_file_list.append(path)
		if 'json_name' in locals():
			tmp_file_list.append(json_name)
		if 'pdf_name' in locals():
			tmp_file_list.append(pdf_name)
		if len(tmp_file_list) > 1:
			file_list.append(tmp_file_list)
	return file_list


def create_xml(df):
	import xml.etree.ElementTree as ET

	for index, row in df.iterrows():
		#print(index, row)
		for item in row:
			#print(item)
			if isinstance(item, str):
				#print("File Name: " + item)
				file_name = item
				output_filename = file_name.replace(".jpeg", ".xml")

				annotation = ET.Element("annotation")
				ET.SubElement(annotation, "folder").text = "VOC2007"
				ET.SubElement(annotation, "filename").text = file_name

				source = ET.SubElement(annotation, "source")
				ET.SubElement(source, "database").text = "The VOC2007 Database"
				ET.SubElement(source, "annotation").text = "PASCAL VOC2007"
				ET.SubElement(source, "image").text = "flickr"
				ET.SubElement(source, "flickrid").text = "341012865"

				owner = ET.SubElement(annotation, "owner")
				ET.SubElement(owner, "flickrid").text = "SCDM"
				ET.SubElement(owner, "name").text = "SCDM Training Data"

			if isinstance(item, list):
				#print("List Name: " + str(item))
				if len(item) == 5:
					x_1 = item[0]
					y_1 = item[1]
					x_2 = item[2]
					y_2 = item[3]
					class_name = item[4]

					#size = ET.SubElement(annotation, "size")
					#ET.SubElement(size, "width").text = "353"
					#ET.SubElement(size, "height").text = "500"
					#ET.SubElement(size, "depth").text = "3"

					ET.SubElement(annotation, "segmented").text = "0"

					img_object = ET.SubElement(annotation, "object")
					ET.SubElement(img_object, "name").text = class_name
					#ET.SubElement(img_object, "pose").text = "Left"
					#ET.SubElement(img_object, "truncated").text = "1"
					ET.SubElement(img_object, "difficult").text = "0"
					bndbox = ET.SubElement(img_object, "bndbox")
					ET.SubElement(bndbox, "xmin").text = str(x_1)
					ET.SubElement(bndbox, "ymin").text = str(y_1)
					ET.SubElement(bndbox, "xmax").text = str(x_2)
					ET.SubElement(bndbox, "ymax").text = str(y_2)
		size = ET.SubElement(annotation, "size")
		ET.SubElement(size, "width").text = str(row[-1])
		ET.SubElement(size, "height").text = str(row[-2])
		#ET.SubElement(size, "depth").text = "3"

		tree = ET.ElementTree(annotation)
		tree.write(os.path.join(input_path, "Annotations", output_filename), xml_declaration=True, encoding='utf-8')


def read_json(json_file):
	import json
	from pprint import pprint

	with open(json_file) as data_file:
		data = json.load(data_file)
	#pprint(data)
	dict = {}

	if 'pdf' in data:
		pdf_name = data.get("pdf")
		print('PDF-NAME: ' + str(pdf_name))

	if 'tags' in data:
		#print('TAGS FOUND!!!')
		#print(data.get("tags"))
		tag_list = data.get("tags")
		for entry in tag_list:
			#print("Entry: " + str(entry))
			for sub_item in entry:
				#print(sub_item)
				if 'name' in sub_item:
					name = entry.get("name")
				elif 'fileName' in sub_item:
					file_name = entry.get("fileName")
				elif 'x' in sub_item:
					x_1 = entry.get("x")
				elif 'y' in sub_item:
					y_1 = entry.get("y")
				elif 'width' in sub_item:
					width = entry.get("width")
				elif 'height' in sub_item:
					height = entry.get("height")
			if 'file_name' in locals():
				if pdf_name.endswith('.pdf'):
					pdf_name = pdf_name.replace('.pdf','')
				elif pdf_name.endswith('.PDF'):
					pdf_name = pdf_name.replace('.PDF','')
				key = pdf_name + '_' + file_name + '_' + name
				dict[key] = [x_1, y_1, width, height, name]
	#print(dict)
	return dict

def create_dataset():
	"""
    extract all raw zip files, create train/test set and annotation files
    :param input_path: The path to the zip folders
    :return: None
    """

	# 1 Unzip folder
	zip_path = os.path.join(input_path, 'Raw/')
	assert os.path.exists(zip_path), 'Path does not exist: {}'.format(zip_path)
	extract_folders(zip_path)

	# 2 Get the name from the pdf file and images
	print(outdir)
	file_list = rename_filenames(outdir)
	print('File list: ' + str(file_list))

	# 3 With 2 + 3 - Create: trainval.txt and test.txt
	renamed_file_list = []
	json_dictionary_list = []

	for file in file_list:
		print('*********** NEXT FILE **********')
		print(file)
		pdf_name = file[-1]
		json_name = file[-2]
		path = file[-3]

		for item in file[:-3]:
			print(item)
			os.rename(os.path.join(path,item), os.path.join(path, item.replace(item, pdf_name + '_' + item)))
			renamed_file_list.append(pdf_name + '_' + item)
			# Move the file to the JPEGImages folder
			os.rename(os.path.join(path, pdf_name + '_' + item), os.path.join(input_path, "JPEGImages", pdf_name + '_' + item))
			# TODO: Append to renamed_file_list without jpeg
		# Read the json file
		json_dictionary = read_json(os.path.join(path, json_name))
		json_dictionary_list.append(json_dictionary)

	print('********* FILE LIST *********')
	print(renamed_file_list)
	#print(json_dictionary_list)

	# Merge the file and json list
	final_list = []
	for file in renamed_file_list:
		#print(file)
		tmp_final_list = []
		tmp_final_list.append(file)
		for item in json_dictionary_list:
			for key, value in item.items():
				#print(key)
				#print(value)
				if file in key:
					#print('KEY FOUND!!!' + str(key))
					tmp_final_list.append(value)
					#break
		final_list.append(tmp_final_list)
	print(final_list)

	# Convert list into pandas dataframe
	df = pd.DataFrame(final_list)
	#	print(df)

	# Iterate over DF and change % to pixel and normalize classes
	size_list = []
	for index, row in df.iterrows():
		#print(index, row)
		for column_index, item in enumerate(row):
			#print(item)
			#print(type(item))
			#print('*****')
			#print(df.iloc[index, column_index])
			#print('*****')
			if isinstance(item, str):
				size = cv2.imread(os.path.join(input_path, "JPEGImages", item)).shape
				height = size[0]
				width = size[1]
				tmp_size_list = []
				tmp_size_list.append(height)
				tmp_size_list.append(width)
				size_list.append(tmp_size_list)
				#print(height, width)
			elif isinstance(item, list):
				#print('List item: ' + str(item))
				# Check if the length is as expected
				if len(item) == 5:
					#print(item[0])
					item[0] = float(item[0].replace("%", ""))*0.01 * width
					#print(item[0])
					#print(item[1])
					item[1] = float(item[1].replace("%", ""))*0.01 * height
					#print(item[1])
					#print(item[2])
					item[2] = item[0] + (float(item[2].replace("%", ""))*0.01 * width)
					#print(item[2])
					#print(item[3])
					item[3] = item[1] + (float(item[3].replace("%", ""))*0.01 * height)
					#print(item[3])
					#print(item[4])
					# Normalize the classes
					if "_" in item[4]:
						item[4] = item[4].split("_", 1)[0]
						#print(item[4])
					tmp_position_list = []
					tmp_position_list.append(item[0])
					tmp_position_list.append(item[1])
					tmp_position_list.append(item[2])
					tmp_position_list.append(item[3])
					tmp_position_list.append(item[4])
					#print(tmp_position_list)
					# Update the data frame
					df.iloc[index, column_index] = tmp_position_list
				else:
					print("ERROR!!!")
	print(df)
	df.to_pickle('meta_dataframe.csv')
	print(size_list)
	df_size = pd.DataFrame(size_list, columns=['Height', 'Width'])
	print(df_size)
	df_merged = pd.concat([df, df_size], axis=1)
	print(df_merged)
	print(df_merged.shape)

	# 5) Create the .xml file for every image
	print('***** PART 5 - CREATE XML FILES *****')
	create_xml(df_merged)

	# 6) Create training and test data set
	# Randomly sample 60% of your dataframe
	print('***** PART 6 - DATASET GENERATION *****')
	#print(df.iloc[:,0])
	#print(df[0])
	new_df = df[0].apply(lambda x: x.replace(".jpeg", ""))
	#new_df.to_csv(os.path.join(input_path,'ImageSets','Main','dataframe.csv'))

	df_trainval = new_df.sample(frac=0.6)
	df_test = new_df.loc[~new_df.index.isin(df_trainval.index)]
	print(df_trainval.shape)
	print(df_test.shape)

	# Write training and test files out
	df_trainval.to_csv(os.path.join(input_path,'ImageSets','Main','trainval.txt'), header=None, index=None, sep=' ', mode='a')
	df_test.to_csv(os.path.join(input_path,'ImageSets','Main','test.txt'), header=None, index=None, sep=' ', mode='a')


def main():
	create_dataset()


if __name__ == '__main__':
	main()