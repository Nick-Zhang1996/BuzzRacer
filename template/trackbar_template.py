
        binary = binary.astype(np.uint8)

        #label connected components
        connectivity = 8 
        #XXX will doing one w/o stats for fast removal quicker?
        output = cv2.connectedComponentsWithStats(binary, connectivity, cv2.CV_32S)
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]


        # apply known rejection standards here
        goodLabels = []
        for i in range(num_labels):
            if (stats[i,cv2.CC_STAT_AREA]<1000 or stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT] < 220 or stats[i,cv2.CC_STAT_HEIGHT]<80):
                binary[labels==i]=0
            else:
                goodLabels.append(i)

        if (len(goodLabels)==0):
            print(' no good feature')
            return
        else:
            print('good feature :  '+str(len(goodLabels)))

# ------- from here basically------
        cv2.namedWindow('binary')
        cv2.imshow('binary',binary)
        cv2.createTrackbar('label','binary',1,len(goodLabels)-1,nothing)
        last_selected = 1

        # visualize the remaining labels
        while(1):

            selected = goodLabels[cv2.getTrackbarPos('label','binary')]

            binaryGB = binary.copy()
            binaryGB[labels==selected] = 0
            testimg = 255*np.dstack([binary,binaryGB,binaryGB])
            cv2.imshow('binary',testimg)

            #list info here

            if (selected != last_selected):
                print('label --'+str(selected))
                print('Area --\t'+str(stats[selected,cv2.CC_STAT_AREA]))
                print('Bottom --\t'+str(stats[selected,cv2.CC_STAT_TOP]+stats[selected,cv2.CC_STAT_HEIGHT]))
                print('Height --\t'+str(stats[selected,cv2.CC_STAT_HEIGHT]))
                print('WIDTH --\t'+str(stats[selected,cv2.CC_STAT_WIDTH]))
                print('---------------------------------\n\n')
                last_selected = selected

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                print('next')
                break
        cv2.destroyAllWindows()
        return
