
from feature.util.enstools_utils import get_vertical_dim, get_longitude_dim, get_latitude_dim
# calc dt and dt^2 (first and second derivative)
def calc_adv(cv, u, v):

    lon_str = get_longitude_dim(cv)
    lat_str = get_latitude_dim(cv)

    dxcv = cv.differentiate(coord=lon_str)  # optional: add edge_order=2. default is 1.
    dycv = cv.differentiate(coord=lat_str)  # optional: add edge_order=2. default is 1.

    adv = -( u *dxcv) - ( v *dycv)           # adv = u*d/dx(cv) + v*d/dy(cv)

    dxadv = adv.differentiate(coord=lon_str)
    dyadv = adv.differentiate(coord=lat_str)

    advadv = ( u *dxadv) + ( v *dyadv)  # to constrain adv to locations where there is a local min.
    return (adv, advadv)
