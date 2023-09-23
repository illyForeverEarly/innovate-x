from collections import OrderedDict
from distutils import dist

import numpy as np


class CentroidTracker:

    def __init__( self, maxDisappeared = 50, maxDistance = 50 ):
        self.nextObjectID = 0 # Initialize next unique object ID
        # Initialize dictionary to keep track of ID-centroid pair
        self.objects = OrderedDict()
        # Initialize dictionary to keep track of "disappeared" IDs
        self.disappeared = OrderedDict()

        # After 50 frames we stop tracking object
        self.maxDisappeared = maxDisappeared

        # To associate an object, store max distance between centroids
        # If distance is bigger than 50, make objects "disappeared"
        self.maxDistance = maxDistance

    # Use nextObjectID to store the centroid during registering object
    def register( self, centroid ):
        self.objects[ self.nextObjectID ] = centroid
        self.disappeared[ self.nextObjectID ] = 0
        self.nextObjectID += 1

    # Delete object's ID from both dictionaries during deregistering
    def deregister( self, objectID ):
        del self.objects[ objectID ]
        del self.disappeared[ objectID ]

    # Check if list of input bounding box rectangles is empty
    def update( self, rects ):
        if len( rects ) == 0:
            # Loop over any existing tracked objects - mark them disappeared
            for objectID in list( self.disappeared.keys() ):
                self.disappeared[ objectID ] += 1
                # If number of consecutive frames with object being missing > 50,
                # then deregister it
                if self.disappeared[ objectID ] > self.maxDisappeared:
                    self.deregister( objectID )

            return self.objects
        
        # Initialize an array of centroids of current frame with zeros
        inputCentroids = np.zeros( ( len(rects), 2 ), dtype = "int" )

        # Loop over the bounding box rectangles
        for ( i, ( startX, startY, endX, endY ) ) in enumerate( rects ):
            # Use bounding box coordinates to derive the centroid
            cX = int( (startX + endX) / 2.0 )
            cY = int( (startY + endY) / 2.0 )
            inputCentroids[i] = (cX, cY) # Element in inputCentroids is a pair

        # If we're not tracking any objects
        if len( self.objects ) == 0:
            for i in range( 0, len( inputCentroids ) ):
                self.register( inputCentroids )

        # Otherwise, try to match input centroids to existing object centroids
        else:
            objectIDs = list( self.objects.keys() )
            objectCentroids = list( self.objects.values() )

            # Euclidian distance: returns 2-D array
            D = dist.cdist( np.array( objectCentroids ), inputCentroids )

            # Find the smallest value in each row and sort it in ascending order
            rows = D.min( axis = 1 ).argsort()

            # Find the smallest value in each column and sort by row
            columns = D.argmin( axis = 1 )[rows] # Returns index of the minimum value

            # Keep track of examined rows and columns
            usedRows = set()
            usedColumns = set()

            # Loop over (row, column) pairs
            for ( row, column ) in zip( rows, columns ):
                # Ignore if examined
                if row in usedRows or column in usedColumns:
                    continue

                # Don't associate two centroids to the same object if the distance
                # between them is greater than maxDistance
                if D[ row, column ] > self.maxDistance:
                    continue

                # Otherwise,
                # Assign a new centroid to objectID of current row
                # Reset disappeared counter
                objectID = objectIDs[row]
                self.objects[ objectID ] = inputCentroids[ column ]
                self.disappeared[ objectID ] = 0

                # Indicate examination of rows and columns
                usedRows.add( row )
                usedColumns.add( column )

            # Compute row and column index that we have NOT yet examined
            unusedRows = set( range( 0, D.shape[0] ) ).difference( usedRows )
            unusedColumns = set( range( 0, D.shape[1] ) ).difference( usedColumns )

            # If number of centroid objects >= number of input centroids,
            # then, check for disappearance
            if D.shape[0] >= D.shape[1]: # If number of rows >= number of columns
                # Loop over unused rows
                for row in unusedRows:
                    # for corresponding's row's object ID increment disappeared counter
                    objectID = objectIDs[ row ]
                    self.disappeared[ objectID ] += 1

                    # Check how long object been disappeared
                    if self.disappeared[ objectID ] > self.maxDisappeared:
                        self.deregister( objectID )

            # Otherwise, number of centroid objects < number of input centroids
            # Register new centroids
            else:
                for column in unusedColumns:
                    self.register( inputCentroids[column] )

            # Return set of trackable objects
            return self.objects


