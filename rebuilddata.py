class CatsVsDogs():
    IMG_SIZE = 50
    CATS = ""
    DOGS = ""
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    j=0
    
    def make_training_data(self):
        for f in tqdm(os.listdir("train")):
            self.j += 1
            if(self.j<12501):
                try:                                     #there might some errors in th pictures
                    path = os.path.join("train",f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                    self.training_data.append([np.array(img),np.eye(2)[0]])               
                except Exception as e:
                    pass
            else:
                try:                                     #there might some errors in th pictures
                    path = os.path.join("train",f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                    self.training_data.append([np.array(img),np.eye(2)[1]])               
                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        
if __name__ == '__main__':
    CatsVsDogs()