from typing import Dict, Any
import utils
import numpy as np
NDArray = Any
import math


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}
    ## WHAT ABOUT NEAREST NEIGHBOURS
    #basic imp + TOO NAIVE
    ##WHEN DO I NEED TO NORMALIZE?
    numpyImage=image
    ## I NEED TO DO image=GRAYSCALE HWO DOES THIS EVEN WORK WITHOUT IT>>???
    heightDelta=image.shape[0]-out_height
    widthDelta=image.shape[1]-out_width
    smallerWidthFlag=(widthDelta>0)
    smallerHeightFlag=(heightDelta>0)
    ##first width, then height
    for t in range(2):
        image=utils.to_grayscale(numpyImage)
        grayscaleImage=utils.to_grayscale(numpyImage)
        imageGradients=utils.get_gradients(numpyImage) ##rotate these instead?
        indexMatrix=np.full((image.shape[0],image.shape[1]),np.arange(image.shape[1]))
        if(t==1):
            numOfSeams=abs(heightDelta)
        if(t==0):
            numOfSeams=abs(widthDelta)

        mMatrix=np.zeros((image.shape[0],image.shape[1]))
        Cv,Cl,Cr=FillCostMatrixForward(0,image,mMatrix,imageGradients,grayscaleImage)
        
        for c in range(numOfSeams):##FOR EACH SEAM:
            print(c)
            mMatrix=np.zeros((image.shape[0], image.shape[1]-c))
            ##calculate cost msatrix do i neede to calc this each time from 0?
            if(forward_implementation and c!=0):
                Cv,Cl,Cr=FixCostMatrix(c,image,mMatrix,imageGradients,grayscaleImage,indexArray) ##THIS TO CHANGE TO NOT NAIVE!
            #else:
                #mMatrix=FillCostMatrixNaive(c,image,mMatrix,imageGradients)
            ##find seam
            minIndex=-1
            minValue=np.Inf
            for i in range(image.shape[1]-c):
                if(mMatrix[image.shape[0]-1][i]<minValue):
                    minValue=mMatrix[image.shape[0]-1][i]
                    minIndex=i
            if(forward_implementation):
                indexArray=FindSeamForward(c,image,minIndex,mMatrix,imageGradients,indexMatrix,Cv,Cl,Cr,grayscaleImage)
            else:
                FindSeamNaive(c,image,minIndex,mMatrix,imageGradients,indexMatrix,grayscaleImage) 
            ##removing rightmost seam
            grayscaleImage = np.delete(grayscaleImage,image.shape[1]-c-1 , 1)
            imageGradients = np.delete(imageGradients,image.shape[1]-c-1 , 1)
            indexMatrix = np.delete(indexMatrix,image.shape[1]-c-1 , 1)            
            ##store the order nad pixels rewmoved in each eiteration DONE
        ##
        ##remove/enlrage by k pixels by smallerFlag 1 for remove 0 for enlrage TODO
        ##return seam painted:
            
        if(numOfSeams==0):
            if(t==0):
                verticalSeams=np.copy(numpyImage)
                numpyImage=RotateImage(numpyImage)
            else:
                numpyImage=RotateBack(numpyImage)
                horizontalSeams=np.copy(numpyImage)
            continue
        
        if(t==0):
            verticalSeams=paintSeams(numpyImage,indexMatrix,t)
            if(smallerWidthFlag):
                numpyImage=RemoveSeams(numpyImage,numOfSeams,indexMatrix)
            else:
                numpyImage=AddSeams(numpyImage,numOfSeams,indexMatrix)
                
            numpyImage=RotateImage(numpyImage)
        if(t==1):
            horizontalSeams=paintSeams(numpyImage,indexMatrix,t)
            if(smallerHeightFlag):
                numpyImage=RemoveSeams(numpyImage,numOfSeams,indexMatrix)
            else:
                numpyImage=AddSeams(numpyImage,numOfSeams,indexMatrix)
                
            numpyImage=RotateBack(numpyImage)
    ##return {'img1':image}
    return { 'resized' : numpyImage, 'vertical_seams' : verticalSeams ,'horizontal_seams' : horizontalSeams}
    ##else:enlarge TODO
    ##image now painted

##indexMatrix[image.shape[0]-1-i,minIndex:-1]=indexMatrix[image.shape[0]-1-i,minIndex+1:]
##indexMatrix = np.delete(indexMatrix,image.shape[1]-c-1 , 1)   
def RemoveSeams2(image,numOfSeams,indexMatrix):
    for i in range(image.shape[0]):
        t=0#counter in line
        for j in range(image.shape[1]-1):##paint seam, NEED C? NEED -1 HERE? did it for the switching
            if(t==indexMatrix.shape[1]):
                break
            if(indexMatrix[i][t]!=j):
                image[i,j:-1]=image[i,j+1:]
            else:
                t=t+1
    for c in range(numOfSeams):
        image = np.delete(image,image.shape[1]-c-1 , 1)
    return image

def RemoveSeams(image,numOfSeams,indexMatrix):
    newImage=np.zeros((image.shape[0], image.shape[1]-numOfSeams,3))
    for i in range(image.shape[0]):
        t=0#counter in line
        for j in range(image.shape[1]):##paint seam, NEED C? NEED -1 HERE? did it for the switching
            if(t==indexMatrix.shape[1]):
                break
            if(indexMatrix[i][t]!=j):
                continue
            else:
                
                newImage[i][t]=image[i][j]
                t=t+1
    return newImage


def AddSeams(image,numOfSeams,indexMatrix):
    newImage=np.zeros((image.shape[0],image.shape[1]+numOfSeams,3))
    for i in range(image.shape[0]):
        t=0#counter in line
        k=0
        for j in range(image.shape[1]):##paint seam, NEED C? NEED -1 HERE? did it for the switching
            if(t==indexMatrix.shape[1]):
                break
            if(indexMatrix[i][t]!=j):
                newImage[i][j+k]=image[i][j]
                newImage[i][j+k+1]=image[i][j]
                k=k+1
            else:
                newImage[i][j+k]=image[i][j]
                t=t+1
    return newImage

def RotateImage(image):
    image=np.rot90(image,k=1,axes=(0,1))
    return image

def RotateBack(image):
    image=np.rot90(image,k=-1,axes=(0,1))
    return image
    




def paintSeams(image,indexMatrix,k):
    ##edges wont paint! #changed this lately from 2 ifs, first has a break, to 2 combined ifs without break!
    print(image.shape[0],image.shape[1],indexMatrix.shape[0],indexMatrix.shape[1])
    paintedImage=np.copy(image)
    for i in range(paintedImage.shape[0]):##changed -1 to not -1 here and in j
        t=0#counter in line
        for j in range(paintedImage.shape[1]):##paint seam, NEED C?
            if(t==indexMatrix.shape[1] or indexMatrix[i][t]!=j):
                paintedImage[i][j][1]=0#color
                paintedImage[i][j][2]=0#color
                if(k==0):
                    paintedImage[i][j][0]=255#color
                else:
                    paintedImage[i][j][0]=0#color
            else:
                t=t+1
    if(k==1):
        return RotateBack(paintedImage)
    return paintedImage

def paintAythingButSeams(image,indexMatrix):
    for i in range(indexMatrix.shape[0]-1):
        t=0#counter in line
        for j in range(indexMatrix.shape[1]-1):##paint seam, NEED C?
            image[i][indexMatrix[i][j]]=0#color
            
    return image
    
def FillCostMatrixNaive(c,image,mMatrix,imageGradients):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]-c):
            mMatrix[i][j]=imageGradients[i][j]
            if (i==0):
                continue
            l=np.Inf
            m=np.Inf
            r=np.Inf
            m=mMatrix[i-1][j]
            if (j!=0):
                l=mMatrix[i-1][j-1]
            if (j!=image.shape[1]-1-c):
                r=mMatrix[i-1][j+1]
            mMatrix[i][j]=mMatrix[i][j]+min(l,m,r)
    return mMatrix      

def FixCostMatrix(c,image,mMatrix,imageGradients,grayscaleImage,indexArray):
    a=grayscaleImage
    E=imageGradients
    zero=np.broadcast_to([0.],[a.shape[0],1])
    zero2=np.broadcast_to([0.],[1,a.shape[1]])
    left=np.concatenate([zero,a[:,0:-1]],axis=1)
    right=np.concatenate([a[:,1:],zero],axis=1)
    up=np.concatenate([zero2,a[::-1]],axis=0)
    up=np.delete(up,a.shape[0],0)

    Cv=np.abs(left-right)
    Cv[:,0]=255
    Cv[:,a.shape[1]-1]=255

    b=np.abs(up-left)
    b[:,0]=255
    b[0,:]=255
    Cl=(Cv+b)

    c=np.abs(up-right)
    c[:,a.shape[1]-1]=255
    c[0,:]=255
    Cr=(Cv+c)
    ## TO DO: IT WORKS NOW. SLOW PART IS up-right, left-right... try to handle this with below double for.
    ## after it works, check how to do it locally in this one below.
    '''
    for i in range(a.shape[0]):
        j=int(indexArray[i][0])##also sides?
        mMatrix[i][j]=E[i][j]
        if(i==0):
            continue
        m=mMatrix[i-1][j]+Cv[i][j]
        l=np.Inf
        r=np.Inf
        if(j!=0):
            l=mMatrix[i-1][j-1]+Cl[i][j]
        if(j!=a.shape[1]-1):
            r=mMatrix[i-1][j+1]+Cr[i][j]
        mMatrix[i][j]=E[i][j]+min(l,m,r)
    '''
    
    for i in range(a.shape[0]):
        mMatrix[i]=E[i]
        if(i==0):
                continue
        m=mMatrix[i-1]+Cv[i]
        l=np.pad(mMatrix[i-1],((1,0)),'constant',constant_values=(np.Inf))
        l2=np.delete(l,l.shape[0]-1,0)
        l3=l2+Cl[i]#1 to the left pad from the left with 255

        r=np.pad(mMatrix[i-1],((0,1)),'constant',constant_values=(np.Inf))
        r2=np.delete(r,0,0)
        r3=r2+Cr[i]#1 to the right pad from the right with 255
        ##do 0 and last alone? 
        for j in range(0, a.shape[1]):
            mMatrix[i][j]=E[i][j]+min(l3[j],m[j],r3[j])

    
    '''
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            mMatrix[i][j]=E[i][j]
            if(i==0):
                continue
            m=mMatrix[i-1][j]+Cv[i][j]
            l=np.Inf
            r=np.Inf
            if(j!=0):
                l=mMatrix[i-1][j-1]+Cl[i][j]
            if(j!=a.shape[1]-1):
                r=mMatrix[i-1][j+1]+Cr[i][j]
            mMatrix[i][j]=E[i][j]+min(l,m,r)

    '''
    return Cv,Cl,Cr
    
    
def FillCostMatrixForward(c,image,mMatrix,imageGradients,grayscaleImage):
    a=grayscaleImage
    E=imageGradients
    zero=np.broadcast_to([0.],[a.shape[0],1])
    zero2=np.broadcast_to([0.],[1,a.shape[1]])
    left=np.concatenate([zero,a[:,0:-1]],axis=1)
    right=np.concatenate([a[:,1:],zero],axis=1)
    up=np.concatenate([zero2,a[::-1]],axis=0)
    up=np.delete(up,a.shape[0],0)
    
    Cv=np.abs(left-right)
    Cv[:,0]=255
    Cv[:,a.shape[1]-1]=255

    b=np.abs(up-left)
    b[:,0]=255
    b[0,:]=255
    Cl=(Cv+b)

    c=np.abs(up-right)
    c[:,a.shape[1]-1]=255
    c[0,:]=255
    Cr=(Cv+c)


    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            mMatrix[i][j]=E[i][j]
            if(i==0):
                continue
            m=mMatrix[i-1][j]+Cv[i][j]
            l=np.Inf
            r=np.Inf
            if(j!=0):
                l=mMatrix[i-1][j-1]+Cl[i][j]
            if(j!=a.shape[1]-1):
                r=mMatrix[i-1][j+1]+Cr[i][j]
            mMatrix[i][j]=E[i][j]+min(l,m,r)


    return Cv,Cl,Cr

def FindSeamNaive(c,image,minIndex,mMatrix,imageGradients,indexMatrix,grayscaleImage):##should pass by ref
    for i in range(image.shape[0]):
        if(i==image.shape[0]-1):
            break#fix
        l=np.Inf
        m=mMatrix[image.shape[0]-2-i][minIndex]
        r=np.Inf
        if (minIndex!=0):
            l=mMatrix[image.shape[0]-2-i][minIndex-1]
        if (minIndex!=image.shape[1]-1-c):
            r=mMatrix[image.shape[0]-2-i][minIndex+1]
        if(mMatrix[image.shape[0]-1-i][minIndex]==imageGradients[image.shape[0]-1-i][minIndex]+r):
            minIndex=minIndex+1
        elif(mMatrix[image.shape[0]-1-i][minIndex]==imageGradients[image.shape[0]-1-i][minIndex]+l):
            minIndex=minIndex-1
        ##minindex stays the same
        grayscaleImage[image.shape[0]-1-i,minIndex:-1]=grayscaleImage[image.shape[0]-1-i,minIndex+1:]##SHIFT TO THE RIGHT
        imageGradients[image.shape[0]-1-i,minIndex:-1]=imageGradients[image.shape[0]-1-i,minIndex+1:]
        indexMatrix[image.shape[0]-1-i,minIndex:-1]=indexMatrix[image.shape[0]-1-i,minIndex+1:]

def FindSeamForward(c,image,minIndex,mMatrix,imageGradients,indexMatrix,Cv,Cl,Cr,grayscaleImage):##should pass by ref
    indexArray=np.zeros((image.shape[0],1))
    indexArray[image.shape[0]-1][0]=minIndex
    for i in range(image.shape[0]):
        
        if(i==image.shape[0]-1):
            break#fix
        l=np.Inf
        m=mMatrix[image.shape[0]-2-i][minIndex]+Cv[image.shape[0]-1-i][minIndex]
        r=np.Inf
        if (minIndex!=0):
            l=mMatrix[image.shape[0]-2-i][minIndex-1]+Cl[image.shape[0]-1-i][minIndex]
        if (minIndex!=image.shape[1]-1-c):
            r=mMatrix[image.shape[0]-2-i][minIndex+1]+Cr[image.shape[0]-1-i][minIndex]
        if(mMatrix[image.shape[0]-1-i][minIndex]==imageGradients[image.shape[0]-1-i][minIndex]+r):
            minIndex=minIndex+1
        elif(mMatrix[image.shape[0]-1-i][minIndex]==imageGradients[image.shape[0]-1-i][minIndex]+l):
            minIndex=minIndex-1
        ##minindex stays the same
        indexArray[image.shape[0]-2-i][0]=minIndex
        grayscaleImage[image.shape[0]-1-i,minIndex:-1]=grayscaleImage[image.shape[0]-1-i,minIndex+1:]##SHIFT TO THE RIGHT
        imageGradients[image.shape[0]-1-i,minIndex:-1]=imageGradients[image.shape[0]-1-i,minIndex+1:]
        indexMatrix[image.shape[0]-1-i,minIndex:-1]=indexMatrix[image.shape[0]-1-i,minIndex+1:]
    return indexArray



    
    

    
