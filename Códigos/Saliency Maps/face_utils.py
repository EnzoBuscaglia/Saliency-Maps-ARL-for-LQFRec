from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt
import itertools
import argparse
from os import listdir, system
import fnmatch
import cv2
import PIL
import torch
import time
from scipy import ndimage
from tqdm.auto import tqdm
#from google.colab.patches import cv2_imshow
import seaborn as sn
from arcface import ArcFace


arcface    = ArcFace.ArcFace()
model_name = 'arcface'
face_model = arcface

def num2fixstr(x,d):
    st = '%0*d' % (d,x)
    return st

def testz(h):
    print('holas ***'+h+'***')


def dirfiles(img_path,img_ext):
    img_names = fnmatch.filter(sorted(listdir(img_path)),img_ext)
    return img_names


def read_img(path):
   img = cv2.imread(path) ## reading image
   (h,w) = img.shape[:2]  ## fetching height and width
   width = 256            ## hard coding width
   ratio = width / float(w) ## preparing a ration for height
   height = int(h * ratio)  ## generating new height
   return cv2.resize(img,(width,height)) ##return the reshaped image


def minmax(X, low, high, minX=None, maxX=None, dtype=np.float):
    X = np.asarray(X)
    if minX is None:
        minX = np.min(X)
    if maxX is None:
        maxX = np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high - low)
    X = X + low
    return np.asarray(X, dtype=dtype)

def extract_hist(img,norm=True):
    hists = []
    num_points = 8
    radii = [1, 2]
    grid_x = 9
    grid_y = 9

    for radius in radii:
        lbp = local_binary_pattern(img,
                                   num_points,
                                   radius, 'nri_uniform')

        height = lbp.shape[0] // grid_x
        width = lbp.shape[1] // grid_y
        indices = itertools.product(range(int(grid_x)),
                                    range(int(grid_y)))
        for (i, j) in indices:
            top = i * height
            left = j * width
            bottom = top + height
            right = left + width
            region = lbp[top:bottom, left:right]
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(region, density=True,
                                   bins=n_bins,
                                   range=(0, n_bins))
            hists.append(hist)

    hists = np.asarray(hists)

    x = np.ravel(hists)
    if norm:
      x = x/np.linalg.norm(x)

    return x


class TanTriggsProc():
    def __init__(self, alpha=0.1, tau=10.0, gamma=0.2, sigma0=2.0, sigma1=3.0):
        self._alpha = float(alpha)
        self._tau = float(tau)
        self._gamma = float(gamma)
        self._sigma0 = float(sigma0)
        self._sigma1 = float(sigma1)

    def compute(self, X):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp

    def extract(self, X):
        X = np.array(X, dtype=np.float32)
        X = np.power(X, self._gamma)
        X = np.asarray(ndimage.gaussian_filter(X, self._sigma1) - ndimage.gaussian_filter(X, self._sigma0))
        X = X / np.power(np.mean(np.power(np.abs(X), self._alpha)), 1.0 / self._alpha)
        X = X / np.power(np.mean(np.power(np.minimum(np.abs(X), self._tau), self._alpha)), 1.0 / self._alpha)
        X = self._tau * np.tanh(X / self._tau)
        return X

tantriggs = TanTriggsProc()


def load_image(model_name,img_path):

  if model_name=='dlib':
    img    = cv2.imread(img_path) ## reading image
    (h,w)  = img.shape[:2]  ## fetching height and width
    width  = 500            ## hard coding width
    ratio  = width / float(w) ## preparing a ration for height
    height = int(h * ratio)  ## generating new height
    img    = cv2.resize(img,(width,height)) ##return the reshaped image
  elif model_name=='lbp':
    img = PIL.Image.open(img_path)
    img = img.convert("L")
    img = img.resize((108, 108), PIL.Image.ANTIALIAS)
    img = np.array(img, dtype=np.uint8)
  elif model_name=='arcface':
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112)) 
  elif model_name=='vggface2':
    img = cv2.imread(img_path)
    img = cv2.resize(img, (160, 160)) 
  elif model_name=='casia_webface':
    img = cv2.imread(img_path)
    img = cv2.resize(img, (160, 160)) 

  return img

def faceimg_features(model_name,img,face_model=None,norm=True):

    if model_name=='lbp':
        #img = img.convert("L")
        #img = img.resize((108, 108), PIL.Image.ANTIALIAS)
        #img = np.array(img, dtype=np.uint8)
        img = tantriggs.extract(img)
        img = minmax(img, 0, 255)
        x   = extract_hist(img,norm=norm)

    elif model_name == 'dlib':
        fl = [[0,len(img)-1,len(img[0])-1,0]]
        x = face_model.face_encodings(img,fl)[0]
        if norm:
            x = x/np.linalg.norm(x)


    elif model_name == 'arcface':
        x = face_model.calc_emb(img) # for !pip install arcface
        #x = face_model.get_feature(img)
        

    elif model_name == 'vggface2':
        timg = torch.tensor(img)
        timg = timg.permute(2, 0, 1)
        timg = (timg - 127.5) / 128.0
        z = timg.unsqueeze(0)
        a = face_model(z)
        x = a.detach().numpy()
        x = x.reshape((512,))

    elif model_name == 'casia_webface':
        timg = torch.tensor(img)
        timg = timg.permute(2, 0, 1)
        timg = (timg - 127.5) / 128.0
        z = timg.unsqueeze(0)
        a = face_model(z)
        x = a.detach().numpy()
        x = x.reshape((512,))
    return x

def process_folder(model_name,origin, target, fmt,face_model=None,show=False):
  l = dirfiles(origin,'*.'+fmt)
  n = len(l)
  print('extracting '+str(n)+' '+model_name+' features...')
  for img_name in tqdm(l):        
      img_path = f'{origin}/{img_name}'
      if show:
          print(img_path)
      feat = facefile_features(model_name,img_path,face_model=face_model)
      np.save(f"{target}/{img_name[:-4]}", feat)

def facefile_features(model_name,img_path,face_model=None):
    img = load_image(model_name,img_path)
    feat = faceimg_features(model_name,img,face_model=face_model)
    return feat

def read_features(path,prefix):
    t = 0
    l = dirfiles(path,prefix+'*.npy')
    n = len(l)
    for np_file in tqdm(l):
      x = np.load(path+'/'+np_file)
      if t==0:
        l = x.shape[0]
        if l==1:
          l = x.shape[1]
        X = np.zeros((n,l))
      X[t,:] = x
      t = t+1
    return X

def gaussian_kernel(size, sigma, type='Sum'):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1,
           -size // 2 + 1:size // 2 + 1]
    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    if type=='Sum':
      kernel = kernel / kernel.sum()
    else:
      kernel = kernel / kernel.max()
    return kernel.astype('double')

def minmax_norm(X):
  Y = X-X.min()
  Y = Y/Y.max()
  return Y


def show_heatmap(Ao,H,ss,alpha=0.5,show=False):
  hm = gaussian_kernel(ss,ss/8.5,type='Max')
  # print(D.max())
  X  = cv2.filter2D(H,-1,hm)

  #X  = X/X.max()
  D  = minmax_norm(X)
  X  = np.uint8(D*255)
  HM = cv2.applyColorMap(X, cv2.COLORMAP_JET)
  Y  = cv2.addWeighted(HM, alpha, Ao, 1-alpha, 0)
  if show:
    G = cv2.cvtColor(X,cv2.COLOR_GRAY2RGB)
    Z = cv2.hconcat([G,Y])
    cv2.imshow('Z',Z)
    #cv2_imshow(Z)
  return D,Y


def ComputeError(N,ix,origin,target,type='Max',show=True,error_Yo='dot'):
  print('reading embeddings...')
  XA  = np.load(target + '/A.npy')
  XB  = np.load(target + '/B.npy')
  n = XB.shape[0]
  XAk = np.zeros((N,n))
  
  for k in tqdm(range(N)):
    if ix[k] == 1:
      XAk[k,:] =  np.load(target + '/A'+num2fixstr(k,5)+'.npy')
    
  Yo = 1
  if error_Yo == 'dot':
    Yo = np.dot(XB,XA)
    
  ik  = ix[0:N]
  S  = np.abs(Yo-np.dot(XB,XAk.T))
  ii = np.argwhere(ik==1)
  S[S>1] = 1
  S[S<0] = 0
  Si  = S[ii]
  if type == 'Max':
    err = Si.max()
  else:
    err = Si.min()
  
  t_op = np.argwhere(S==err)


  X = cv2.imread(origin+'/A' +num2fixstr(t_op,5) + '.jpg')
  # print('Best: '+str(t)+'   Error = '+str(err))
  if show:
    cv2.imshow('X',X)
    #cv2_imshow(X)
  return X,S,t_op,err

def MaskedImages(A,s,d,gauss_min,origin,Ao,type='Remove'):


  height    = A.shape[0]
  width     = A.shape[1]


  H2 = height+2*s;
  W2 = width+2*s;
  nh = range(0,H2-s,d)
  nw = range(0,W2-s,d)
  N  = len(nh)*len(nw)
  xy = np.zeros((N,2))
  h  = 1-gaussian_kernel(s,s/8.5,type='Max')
  k  = 0
  xo = np.ones((H2,W2))
  print('computing ' + str(N) + ' masks...')

  if type == 'Remove':
    remove = True
  else:
    remove = False
    AA = 255 - A

  for i in tqdm(range(0,W2-s,d)):
    for j in range(0,W2-s,d):
      x = np.ones((H2,W2))
      x[i:i+s,j:j+s] = h
      xk = x[s:s+height,s:s+width]
      M = np.zeros((height,width,3))
      if xk.min()<gauss_min:
        M[:,:,0] = xk
        M[:,:,1] = xk
        M[:,:,2] = xk
        xymin = np.unravel_index(np.argmin(xk, axis=None), xk.shape)
        xy[k,:] = (xymin[1],xymin[0])
        if remove:
          Ak = np.multiply(M,A)
        else:
          An = (255-np.multiply(1-M,Ao))/255
          Ak = 255 - np.multiply(AA,An)
        st  = origin+'/A' +num2fixstr(k,5)+'.jpg'
        cv2.imwrite(st,Ak)
        k = k+1
  np.save('xy',xy)

  N = k
  print(str(N) + ' masked images generated (from A00000 to A'+num2fixstr(N-1,5)+'.jpg)')

  return xy,N


import os
def PrepareData(origin,target,fileA,fileB,height,width,show=True):
  system('rm -rf '+ origin)
  system('rm -rf '+ target)
  system('mkdir ' + origin)
  system('mkdir  '+ target)
  A = cv2.imread(fileA)
  A = cv2.resize(A, (width,height))
  Ao = A
  cv2.imwrite(origin +'/A.jpg',A)
  B = cv2.imread(fileB)
  B = cv2.resize(B, (width,height))
  cv2.imwrite(origin+'/B.jpg',B)
  if show:
    print('Face-A, Face-B')
    C = cv2.hconcat([A,B])
    cv2.imshow('imagen',C)
    #cv2_imshow(C)
  return A,B


def saliency_1(Ao,B,p,model_name,origin, target, fmt,face_model=None,show=False):
    print('computing saliency-1...')
    ss        = p[0]
    th        = p[1]
    steps     = p[2]
    min_gauss = p[3]

    A         = Ao
    height    = A.shape[0]
    width     = A.shape[1]
    ix        = np.ones((50000,1))
    xy,N      = MaskedImages(A,ss,steps,min_gauss,origin,Ao)

    process_folder(model_name,origin, target, 'jpg',face_model=face_model)
    X,S,t,err = ComputeError(N,ix,origin,target,type='Max',show=show)
    D = np.zeros((height,width))
    for i in range(N):
        if S[i]>th:
            D[int(xy[i,1]),int(xy[i,0])]=S[i]

    return D
  
def saliency_2(Ao,B,p,model_name,origin, target, fmt,face_model=None,show=False):  
    print('computing saliency-2...')
    ss        = p[0]
    th        = p[1]
    steps     = p[2]
    min_gauss = p[3]

    A    = Ao
    err  = 0
    k    = 0
    height = A.shape[0]
    width  = A.shape[1]
    D    = np.zeros((height,width))
    ix   = np.ones((50000,1))
    while err<th:
        k = k+1
        print('iteration: '+str(k)+'...')
        xy,N = MaskedImages(A,ss,steps,min_gauss,origin,Ao)
        process_folder(model_name,origin, target, 'jpg',face_model=face_model)
        X,S,t,err = ComputeError(N,ix,origin,target,type='Max',show=show)
        i = int(xy[t,1])
        j = int(xy[t,0])
        print('D[' + str(i)+ ',' +str(j) + '] = '+str(err))
        D[i,j]=1-err # see error in saliency_minus
        ix[t] = 0
        A = X
    return D


def saliency_3(Ao,B,p,model_name,origin, target, fmt,face_model=None,show=False):  
    print('computing saliency-3...')
    ss        = p[0]
    th        = p[1]
    steps     = p[2]
    min_gauss = p[3]


    height = B.shape[0]
    width  = B.shape[1]
    A    = np.random.rand(height,width,3)
    err  = 1
    k    = 0
    D    = np.zeros((height,width))
    ix   = np.ones((100000,1))
    while err>th:
        k = k+1
        print('iteration: '+str(k)+'...')
        xy,N = MaskedImages(A,ss,steps,min_gauss,origin,Ao,type='Add')
        process_folder(model_name,origin, target, 'jpg',face_model=face_model)
        X,S,t,err = ComputeError(N,ix,origin, target,type='Min',show=show)
        ix[t] = 0
        D[int(xy[t,1]),int(xy[t,0])]=err
        A = X
    return D


def saliency_minus(Ao,B,p,model_name,origin, target, fmt,face_model=None,show=False,error_Yo='dot'):  
    print(' ')
    print('computing saliency-minus:')
    print('=========================')

    ss        = p[0]
    th0       = p[1]
    steps     = p[2]
    min_gauss = p[3]
    kmax      = p[4]
    th1       = p[5]

    A    = Ao
    err  = 0
    k    = 0
    height = A.shape[0]
    width  = A.shape[1]
    H    = np.zeros((height,width))
    ix   = np.ones((50000,1))
    err0 = 1
    while err<th1 and k<kmax: 
        k = k+1
        print(' ')
        print('> minus iteration: '+str(k)+'...')
        xy,N = MaskedImages(A,ss,steps,min_gauss,origin,Ao)
        process_folder(model_name,origin, target, 'jpg',face_model=face_model)
        X,S,t,err = ComputeError(N,ix,origin,target,type='Max',show=show,error_Yo=error_Yo)
        
        if k==1: # for first iteration
          Ho = np.zeros((height,width))
          for i in range(N):
            if S[i]>th0:
              Ho[int(xy[i,1]),int(xy[i,0])]=S[i]

        i   = int(xy[t,1])
        j   = int(xy[t,0])
        st  = "{:.2f}".format(err)
        st0 = "{:.2f}".format(err0)
        st1 = "{:.2f}".format(err0-err)

        print('err = '+st +': H[' + str(i)+ ',' +str(j) + '] = '+st0+'-'+st+' = '+st1)
        err0 = err0-err
        H[i,j]=err0
        ix[t] = 0
        A = X
    return Ho,H


def saliency_plus(Ao,B,p,model_name,origin, target, fmt,face_model=None,show=False,error_Yo='dot'):  
    print(' ')
    print('computing saliency-plus:')
    print('=========================')

    ss        = p[0]
    th0       = p[1]
    steps     = p[2]
    min_gauss = p[3]
    kmax      = p[4]
    th1       = p[5]
    height    = B.shape[0]
    width     = B.shape[1]

    A    = np.random.rand(height,width,3)
    err  = 1
    k    = 0
    H    = np.zeros((height,width))
    ix   = np.ones((50000,1))
    err0 = 1
    while err>th1 and k<kmax:
        k = k+1
        print(' ')
        print('> plus iteration: '+str(k)+'...')

        xy,N = MaskedImages(A,ss,steps,min_gauss,origin,Ao,type='Add')
        process_folder(model_name,origin, target, 'jpg',face_model=face_model)
        X,S,t,err = ComputeError(N,ix,origin, target,type='Min',show=show,error_Yo=error_Yo)

        if k==1: # for first iteration
          Ho = np.zeros((height,width))
          for i in range(N):
            if S[i]>th0:
              Ho[int(xy[i,1]),int(xy[i,0])]=1-S[i]

        i   = int(xy[t,1])
        j   = int(xy[t,0])
        st  = "{:.2f}".format(err)
        st0 = "{:.2f}".format(err0)
        st1 = "{:.2f}".format(err0-err)

        print('err = '+st +': H[' + str(i)+ ',' +str(j) + '] = '+st0+'-'+st+' = '+st1)
        H[i,j]=err0-err
        #H[i,j]=1-err
        err0 = err
        ix[t] = 0
        A = X
    return Ho,H



def match_score(model_name,img_path1, img_path2,face_model=None):
  x1 = facefile_features(model_name,img_path1,face_model=face_model)
  x2 = facefile_features(model_name,img_path2,face_model=face_model)
  sc = np.dot(x1,x2)
  return sc



def face_matches(X,Y,show=True,annot=False):
    Z = np.dot(X,Y.T)
    if show:
      sn.heatmap(Z, annot=annot)



def show_contours(D,A,color_map,contour_levels,print_levels=False,color_levels=True):

    # D: heatmap
    # A: background image
    # Examples
    # show_contours(D,A,'jet'  ,10,print_levels=False,color_levels=True,img_file='Cont.png')
    # show_contours(D,A,'white',10,print_levels=True,color_levels=False,img_file=None)

    height    = D.shape[0]
    width     = D.shape[1]
    levels    = np.linspace(0.1, 1.0, contour_levels)
    x         = np.arange(0, width, 1)
    y         = np.arange(0, height, 1)
    extent    = (x.min(), x.max(), y.min(), y.max())

    Z = D/D.max()
    At = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(At,extent=extent)
    if color_levels:
      CS = plt.contour(Z, levels, cmap=color_map, origin='upper', extent=extent)
    else:
      CS = plt.contour(Z, levels, colors=color_map, origin='upper', extent=extent)

    if print_levels:
      plt.clabel(CS,fontsize=9, inline=1)
    plt.axis('off')
    plt.savefig ('Contour.png',bbox_inches = 'tight',pad_inches = 0)
    # plt.show()
    plt.clf()
    C = cv2.imread('Contour.png')
    C = cv2.resize(C,(width,height))
    return C

def howis(X):
  print('size = '+str(X.shape))
  print('min  = '+str(X.min()))
  print('max  = '+str(X.max()))

def saliency_minusFast(A,B,nh,d,n,nmod,th):
    N    = A.shape[0]
    M    = A.shape[1]  
    H1   = np.zeros((N,M))
    xA   = face_embedding(A)
    xB   = face_embedding(B)
    sc0  = np.dot(xA,xB)
    st   = "{:7.4f}".format(sc0)
    print('minus t = 000 sc[0]='+st)
    #imshow(cv2.hconcat([A,B]),show_pause=1,title='sc0 = '+st)
    t    = 0
    sct  = sc0
    Bt   = B
    dsc  = 1
    t0   = time.time()
    while dsc>th and t<n: # and (t==0 or sct>0): 
        t = t+1
        Hsc,out_min,__ = removing_score(A,Bt,d,nh,ROI=Bt,background=sc0)
        if t==1:
            H0 = sc0-Hsc
        sct     = out_min[1]       # minimal score by removing a gaussian mask centered in (i,j)
        i       = out_min[2]
        j       = out_min[3]
        sct_st  = "{:7.4f}".format(sct)
        dsc     = sc0-sct
        H1[i,j] = dsc
        sc0 = sct
        hij_st = "{:.4f}".format(H1[i,j])
        t1 = time.time()  
        dt_st = "{:.2f}".format(t1-t0)
        t0     = t1  
        print('minus t = '+num2fixstr(t,3)+' sc[t]='+sct_st+ ' H[' + num2fixstr(i,3)+ ',' +num2fixstr(j,3) + '] = sc[t-1]-sc[t] = '+hij_st+' > '+dt_st+'s')
        Bt      = out_min[0]
        #if np.mod(t,nmod)==0:
        #    imshow(Bt,show_pause=1)
    return H0,H1


def face_embedding(img):

    if model_name=='lbp':
        imgr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgr = cv2.resize(imgr,(108,108)) ##return the reshaped image
        x   = extract_hist(imgr)
    elif model_name == 'dlib':
        fl = [[0,len(img)-1,len(img[0])-1,0]]
        x = face_model.face_encodings(img,fl)[0]
        x = x/np.linalg.norm(x)
    elif model_name == 'vggface2':
        timg = torch.tensor(img)
        timg = timg.permute(2, 0, 1)
        timg = (timg - 127.5) / 128.0
        z = timg.unsqueeze(0)
        a = face_model(z)
        x = a.detach().numpy()
        x = x.reshape((512,))
    elif model_name == 'casia_webface':
        timg = torch.tensor(img)
        timg = timg.permute(2, 0, 1)
        timg = (timg - 127.5) / 128.0
        z = timg.unsqueeze(0)
        a = face_model(z)
        x = a.detach().numpy()
        x = x.reshape((512,))
    if model_name == 'arcface':
        x = face_model.calc_emb(img)
    return x

def imshow(I,height=6,width=None,show_pause=0,title=None,fpath=None):
  n = height
  if width!=None:
    m = width
  else:
    N = I.shape[0]
    M = I.shape[1]
    m = round(n*M/N)
  __,ax = plt.subplots(1,1,figsize=(m,n))
  bw = len(I.shape)==2
  if bw:
     ax.imshow(I)
  else:
     ax.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
  if title!=None:
    ax.set_title(title)
  plt.axis('off')
  if fpath!=None:
     plt.savefig (fpath,bbox_inches = 'tight',pad_inches = 0)
  if show_pause>0:
    plt.pause(show_pause)
    plt.close()
  else:
    plt.show()

def removing_score(A,B,d,nh,ROI=None,background=1.0):
  N      = A.shape[0]
  M      = A.shape[1]
  xA     = face_embedding(A)
  Hsc    = background*np.ones((N,M))
  sc_min = 10
  sc_max = -10
  if ROI is None:
      ROI = np.ones((N,M,3))
  #if len(ROI.shape)==2:
  #    ROI = np.dstack((ROI,ROI,ROI))
  for ic in range(d,N,d):
    for jc in range(d,M,d):
      if np.sum(ROI[ic,jc,:])>0:
        Mk  = DefineGaussianMask(ic,jc,nh)
        Bk  = MaskMult(B,Mk)
        xBk = face_embedding(Bk)
        sc  = np.dot(xA,xBk)
        Hsc[ic,jc] = sc
        if sc<sc_min:
            sc_min  = sc
            out_min = [Bk,sc_min,ic,jc]
        if sc>sc_max:
            sc_max  = sc
            out_max = [Bk,sc_max,ic,jc]
  return Hsc,out_min,out_max

def DefineGaussianMask(ic,jc,nh,N=256,M=256):
  # Define an image of NxM, with a Gaussian of nh x nh centred in (ic,jc)
  nh2 = round(nh/2)
  i1  = ic
  j1  = jc
  n   = N+nh
  m   = M+nh
  s   = nh/8.5
  h   = 1-gaussian_kernel(nh,s,type='Max')
  Mk  = np.ones((n,m))
  i2  = i1+nh
  j2  = j1+nh
  Mk[i1:i2,j1:j2] = h
  return Mk[nh2:nh2+N,nh2:nh2+M]

def MaskMult(A,Mk):
    n = Mk.shape[0]
    m = Mk.shape[1]
    M = np.zeros((n,m,3))
    M[:,:,0] = Mk
    M[:,:,1] = Mk
    M[:,:,2] = Mk
    Ak = np.multiply(M,A)
    return Ak.astype(np.uint8)
