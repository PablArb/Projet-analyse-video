from Modules import np
from SettingsConstructor import Settings


def createRoundMask(r: int):
    maskSize = 2*r+1
    center = r
    X, Y = np.ogrid[:maskSize, :maskSize]
    distances = np.sqrt( (X-center)**2 + (Y-center)**2 )
    mask = np.zeros((maskSize, maskSize))
    mask[distances <= r] = 1
    return mask     


def createPinnedMarkerIndicator(image: np.array, position: tuple, r: int):
    mask = createRoundMask(r)
    xmin, ymin ,xmax ,ymax = position[1]-r, position[0]-r, position[1]+r+1, position[0]+r+1
    image[xmin:xmax, ymin:ymax][mask == True] = [0, 255, 0]
    return image


def drawPinnedMarkersIndicators(image: np.array, L, radius) -> np.array:
    MarkedImage = np.copy(image)
    for pinnedMarker in L:
        MarkedImage = createPinnedMarkerIndicator(image, pinnedMarker.coordinates, radius)
    return MarkedImage
