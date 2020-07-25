import os, shutil

import pydicom
from png import Writer

# library for data manipulation
import pandas as pd
import numpy as np

# keras library converting images to tensor
from keras.preprocessing.image import ImageDataGenerator

# decorated iterator
from tqdm import tqdm


class Dicomutils:
    """
    Class for data preparation for dicom files
    """
    
    # constructor
    def __init__(self,src_path,des_path=None):
        """
        Constructor accepts source path as the argument
        """
        self.src_path = src_path
        self.des_path = des_path
        
    
    def read_dicom_file(self,patient_id):
        """
        Function to read dicom file and returns dicom object
        
        Args:
            patient_id = patient id in the dicom file
            source_path = source path for dicom file
        
        Returns:
            dicom object 
        """
        # handling exception for file not found
        try:
            print('patient id:',patient_id)
            file_path = os.path.join(self.src_path,patient_id+'.dcm')
            print('file path: ',file_path)
            dc = pydicom.dcmread(file_path)
            return dc
        except:
            return print('file not found!!')
        

    def get_metadata(self,file_name):
        """
        Function to return metadata(patient age & sex) required

        Args:
            file_name: dicom file name
            s_path: path of the source files
        Returns:
            list containing patient data
        """
        # incase of file not found: returns no object
        try:
            ds = self.read_dicom_file(file_name)
            
            # list to carry patient info
            patient_info = []
            # dicom age element was in str and below slice is used only to take the age of
            # patient. i.e single character is represented by single digit
            #print('Patient age = {0} & sex ={1}'.format(ds["PatientAge"][:3],ds["PatientSex"][0]))
            patient_age = ds["PatientAge"][:3]
            patient_sex = ds["PatientSex"][0]
            
            # adding patient age and sex into to list
            patient_info = [patient_age,patient_sex]
            # returning the list containing patient info
            return patient_info

        except: 
            patient_info =[np.NaN,np.NaN]

        return patient_info
    
    
    """
    This function needs to be put inside thread for multiprocessing....
    """
    
    # image conversion function:
    def extract_patient_metadata(self,patient_files,valid=False):
        """
        Function to extract metadata(Patient_Id, Age, Sex) from the dicom
        file and return the final csv file

        Args:
            source_path = directory containing dicom files
            destination_path = directory to holder the converted image files
            patient_files = list containing patient id

        Returns:
            doesn't return anything, only outputs csv file with patient record
        """
        
        # creating dataFrame with patient info column
        cols =['Patient_Id','Age','Sex']
        # creating empty dataframe with only columns name
        patient_df = pd.DataFrame(columns=cols)
        # enter data with patient id
        patient_df['Patient_Id'] = patient_files
        
        print('Process started...')
        
        
        if self.des_path == None:
            if valid == False:
            # create folder in the source path
                d_path = os.path.join(self.src_path,'train')
            else:
                d_path = os.path.join(self.src_path,'valid')
        else: 
            if valid == False:
            # create folder in the destination path
                d_path = os.path.join(self.des_path,'train')
            else:
                d_path = os.path.join(self.des_path,'valid')
            
        # create a folder to store the converted images
        try:
            #creating directory if not exists
            os.makedirs(d_path)
            folder_name = d_path.split('/')[-1].split('\\')[-1]
            print(f"{folder_name} directory created!!")
        except OSError:
            print('Directory already exists!!!')
            
        
        # for printing progress bar 
        progress_bar = tqdm(total=len(patient_files))
        #patient_files is the list containing all the patient ids
        for patient_id in patient_files:
            
            # calling function to get the meta data
            result = self.get_metadata(patient_id)

            # calling function to convert the image  
            self.dicom_to_png(d_path,patient_id)

            # fills the age and sex column related to the patient 
            # as returned by the function 
            patient_df.loc[patient_df['Patient_Id'] == patient_id,'Age'] = result[0]
            patient_df.loc[patient_df['Patient_Id'] == patient_id,'Sex'] = result[1]
            
            progress_bar.update(1)
            
        progress_bar.close()
        
        # checking if data entered into dataFrame properly
        if patient_df.empty:
            print("DataFrame is empty: Smthing went wrong!!!")
        else:
            print('Writing the dataFrame to csv file')
            patient_df.to_csv(os.path.join(self.src_path,'patient_records.csv'),
                              index=False)
            print('File conversion completed...')
        
        
    def dicom_to_png(self,d_path,file_name):
        """
        Function to convert the pixel into image format and place in
        the destination directory

        Args:
            source_path: files source directory
            des_path: files destination directory
            file_name: patient id

        Returns:
            metadata related to patient
        """

      # incase no file found: returns null object
        try:
            # read and parse using read_dicom function
            ds = self.read_dicom_file(file_name)
            # for image pixel to image format
            shape = ds.pixel_array.shape
            # Convert to float to avoid overflow or underflow losses
            image_2d = ds.pixel_array.astype('float32')

            # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

            # convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)
        except:
            return print('No file to write..')
    
        try:
            # patient id for the image file name as required by later process
            #file_name = ds["PatientID"]

            # Write the PNG file
            with open(os.path.join(d_path,file_name+'.png'),'wb') as png_file:
                #print('Converting pixel to image started..')
                w = Writer(shape[1],shape[0],greyscale=True)
                w.write(png_file,image_2d_scaled)
        except:
            # use proper exception msg
            return print('Errro while writing the file!!!')   
    
      # utility code to seperate the training & validation file
      # as we already have extracted the image from the dicom
      # file best possible way now is to seperate the files
      # into respective folders


#     # function to move files
#     def move_files(self,files,dst_folder):
#         """
#         Function to move from source to destination

#         Args:
#             src_path:source directory
#             dst_folder: destination folder

#         Returns:
#             None
#         """

#         des_path = join_path(src_path,dst_folder)
#         try:
#             # creating directory if not exists
#             os.makedirs(des_path)
#             #os.makedris(dst/valid)
#             print('directory created!!!')
#         except OSError:
#             print('Directory already exists!!!')

#         # moving files
#         for f in files:
#             try:
#                 #print(des_path)
#                 shutil.move(join_path(self.src_path,f),des_path)
#             except:
#                 print('file not found or already moved')
#                 pass

#           # function which only returns common files on the patient records and files in
#           # the directory as we want specific files to be moved
#           #Note: os.listdire lists both sub-directories as well as files in the folder

#     def common_files(path,df):

#         # listing all the contain of the directory 
#         dir_files = os.listdir(path)
#         # listing all the patientId in the dataFrame and concat '.png' extension
#         # for comparison later
#         files = [file+'.png' for file in df['patientId']]
#         # take only the files common on both the above list
#         com_files = list(set(files).intersection(dir_files))

#         return com_files

#     # creating folder with number of classese and moving the files

#     # function move the files to respective class folder
#     def class_folder(path,classes,df):
#         """"
#         Function to move files to respective class folder

#         Args:
#             path: files path
#             classes: array containing the classes
#             df: dataFrame

#         Returns:
#             None
#         """
            
#         for c in classes:
#             temp_df = df.loc[df['Target'] == c]
   
#             # using common_files method to list only the common files
#             files = common_files(path,temp_df)
#             # using move files method to move the files to class folder
#             # wrapping the int c into str, as required
#             move_files(path,files,str(c))
#             #print(files[:10])
           
    