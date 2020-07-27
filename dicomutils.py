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
    __base_path = os.getcwd()
    
    # constructor
    def __init__(self,src_path,des_path=None):
        """
        Constructor accepts source path as the argument
        
        Args:
            src_path: source path for the files
            des_path: destination path for the files
                        if None then same as src_path
        """
        self.src_path = src_path
        self.des_path = des_path
        
    
    def read_dicom_file(self,patient_id):
        """
        Function to read dicom file and returns dicom object
        
        Args:
            patient_id = patient id in the dicom file
        
        Returns:
            dicom object 
        """
        # handling exception for file not found
        try:
            #print('patient id:',patient_id)
            file_path = os.path.join(self.src_path,patient_id+'.dcm')
            #print('file path: ',file_path)
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
    def extract_patient_data(self,train,valid):
        """
        Function to extract metadata(Patient_Id, Age, Sex) from the dicom
        file and return the final csv file

        Args:
            train = training data
            valid = validation data

        Returns:
            doesn't return anything, only outputs csv file with patient record
        """
        
        # creating dataFrame with patient info column
        cols =['patientId','Age','Sex']
        # creating empty dataframe with only columns name
        patient_df = pd.DataFrame(columns=cols)
        # enter data with patient id
        patient_df['patientId'] = train
        
        # adding id of validation data as later needed
        valid_series = pd.Series(valid)
        patient_df['patientId'].append(valid_series,ignore_index=True)
        
        
        print('Process started...')       
        if self.des_path == None:
            train_path = os.path.join(self.src_path,'train')
            valid_path = os.path.join(self.src_path,'valid')
        else: 
            train_path = os.path.join(self.des_path,'train')
            valid_path = os.path.join(self.des_path,'valid')
            
        # create a folder to store the converted images
        try:
            #creating directory if not exists
            os.makedirs(train_path)
            os.makedirs(valid_path)
            train_folder = train_path.split('/')[-1].split('\\')[-1]
            valid_folder = valid_path.split('/')[-1].split('\\')[-1]
            print(f"{train_folder}, {valid_folder} directories created!!")
        except OSError:
            print('Directories already exists!!!')
            
        
        # for printing progress bar 
        print('converting training files')
        progress_bar = tqdm(total=len(train))
        #patient_files is the list containing all the patient ids
        for patient_id in train:
            
            # calling function to get the meta data
            result = self.get_metadata(patient_id)

            # calling function to convert the image  
            self.dicom_to_png(train_path,patient_id)

            # fills the age and sex column related to the patient 
            # as returned by the function 
            patient_df.loc[patient_df['Patient_Id'] == patient_id,'Age'] = result[0]
            patient_df.loc[patient_df['Patient_Id'] == patient_id,'Sex'] = result[1]
            
            progress_bar.update(1)
            
            
        progress_bar.close()
        
         # for printing progress bar 
        print('converting validation files')
        progress_bar = tqdm(total=len(valid))
        #patient_files is the list containing all the patient ids
        for patient_id in valid:
            
            # calling function to get the meta data
            result = self.get_metadata(patient_id)

            # calling function to convert the image  
            self.dicom_to_png(valid_path,patient_id)

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
            patient_df.to_csv(os.path.join(self.__base_path,'patient_records.csv'),
                              index=False)
            print('File conversion completed...')
        
        
    def dicom_to_png(self,d_path,file_name):
        """
        Function to convert the pixel into image format and place in
        the destination directory

        Args:
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
    

    # private function to move files
    def __move_files(self,src_path,dst_folder,files):
        """
        Function to move from source to destination

        Args:
            src_path:source directory
            dst_folder: destination folder
            files: name of files

        Returns:
            None
        """
              
        des_path = os.path.join(src_path,dst_folder)
        try:
            # creating directory if not exists
            os.makedirs(des_path)
            #os.makedris(dst/valid)
            print('directory created!!!')
        except OSError:
            print('Directory already exists!!!')

        # progress bar for moving files
        pbar = tqdm(total=len(files))
        # moving files
        for f in files:
            #print(os.path.join(src_path,f))
            try:
                print(des_path)
                shutil.move(os.path.join(src_path,f),des_path)
                # update progress bar
                
            except:
                print('file not found or already moved')
                pass
            
            pbar.update(1)
            

          # function which only returns common files on the patient records and files in
          # the directory as we want specific files to be moved
          #Note: os.listdire lists both sub-directories as well as files in the folder

    def __common_files(self,path,df):

        # listing all the contain of the directory 
        dir_files = os.listdir(path)
        # listing all the patientId in the dataFrame and concat '.png' extension
        # for comparison later
        files = [file+'.png' for file in df['patientId']]
        # take only the files common on both the above list
        com_files = list(set(files).intersection(dir_files))

        return com_files

    # creating folder with number of classese and moving the files

    # function move the files to respective class folder
    def class_folder(self,df,path,classes):
        """"
        Function to move files to respective class folder

        Args:
            df: dataFrame
            path: files path
            classes: array containing the classes

        Returns:
            None
        """
            
        for c in classes:
            temp_df = df.loc[df['Target'] == c]
   
            # using common_files method to list only the common files
            files = self.__common_files(path,temp_df)
            #print(files)
            # using move files method to move the files to class folder
            # wrapping the int c into str, as required
            self.__move_files(path,str(c),files)