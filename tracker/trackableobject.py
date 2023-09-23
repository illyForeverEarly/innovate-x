class TrackableObject:

    def __init__( self, objectID, centroid ):
         # Store object ID
         # Initialize list of centroids
         self.objectID = objectID
         self.centroids = [ centroid ]

         #Initialize bool to check if object was counted
         self.counted = False