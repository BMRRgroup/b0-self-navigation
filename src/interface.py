import numpy as np
import os
import h5py as h5
import copy
from hmrGC.dixon_imaging.helper import calculate_pdff_percent
from hmrGC.dixon_imaging import MultiEcho
from scipy.ndimage import binary_fill_holes
import pickle
import hdf5storage
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
plt.style.use('seaborn-colorblind')

class ImDataParamsBMRR():
    def __init__(self, filename, dicom=False):        
        self.ImDataParams = {}
        self.AlgoParams = {}
        self.VARPROparams = {}
        self.WFIparams = {}
        self.MotionParams = {} 
        self.Masks = {}
        if dicom:
            self._load_dicom(filename)
        else:
            if filename[-3:] == '.h5':
                self._load_h5(filename)
        self.Masks['airSignalThreshold_percent'] = 5

    def _load_h5(self, filename):
        """load a "*ImDataParamsBMRR.h5" file
        and write into ImDataParamsBMRR object
        :param filename:
        """
        print(f"Load {filename} ...", end="")
        with h5.File(filename, 'r') as f:
            nest_dict = recursively_load_attrs(f, load_data=True)

        attrs = nest_dict.keys()
        for attr in attrs:
            try:
                params = getattr(self, attr)
            except AttributeError:
                setattr(self, attr, {})
                params = getattr(self, attr)
            params.update(nest_dict[attr])
        if "filename" not in self.ImDataParams.keys():
            self.ImDataParams["filename"] = filename
        if isinstance(self.ImDataParams["fileID"], bytes):
            self.ImDataParams["fileID"] = self.ImDataParams["fileID"].decode('utf-8')
        print("Done!")

    def load_WFIparams(self, filename):
        """load a "*WFIparams.mat" file save from MATLAB
        and return it as a python dict
        :param filename:
        :returns:
        :rtype:
        """
        _, file_extension = os.path.splitext(filename)
        if file_extension == '.mat':
            print(f'Load {filename}... ', end='')
            with h5.File(filename, 'r') as f:
                attrs_dict = recursively_load_attrs(f)
                data_dict = recursively_load_data(f, attrs_dict)

            nested_dict = nest_dict(data_dict)

            self.WFIparams = nested_dict['WFIparams']
            self.set_T2s_ms()
            self.set_fatFraction_percent()

            print('Done.')
        elif file_extension == '.pickle':
            with open(filename, 'rb') as f:
                self.WFIparams = pickle.load(f)

    def save_WFIparams(self, savename=None, mat_file=True):
        self._save_params(params2save='WFIparams', savename=savename, mat_file=mat_file)

    ## water-fat separation
    def set_FatModel(self, name='default'):
        if name == 'default':
            self.AlgoParams['FatModel'] = {'freqs_ppm': np.array([-3.8 , -3.4 , -3.1 , -2.68, -2.46, -1.95, -0.5 ,  0.49,  0.59]),
                                           'relAmps': np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 
                                                                0.01498501, 0.03996004, 0.00999001, 0.05694306]),
                                           'name': 'Ren marrow'}
        elif name == 'Peanut oil':
           self.AlgoParams['FatModel'] = {'freqs_ppm': np.array([-3.8 , -3.4 , -3.11, -2.67, -2.45, -1.93, -0.6 , -0.4 ,  0.51, 0.61]),
                                          'relAmps': np.array([0.08749757, 0.56387323, 0.05833171, 0.09605289, 0.05833171, 0.01963834, 
                                                            0.0194439 , 0.0194439 , 0.00972195, 0.06766479]),
                                          'name': 'Peanut oil'}

    def get_tissueMask(self, threshold=None, min_echo=False,
                       single_object=False, wfi_image=False, iDyn=None):
        """set tissue mask based on thresholding the MIP"""
        if threshold is None:
            threshold = self.Masks['airSignalThreshold_percent']
        if wfi_image != False:
            signal = self.WFIparams[wfi_image]
        else:
            if iDyn != None:
                signal = self.ImDataParams['signal'][:,:,:,iDyn,:]
            else:
                signal = self.ImDataParams['signal']
        if min_echo:
            echoMin = np.min(np.abs(signal), axis=3)
            tissueMask = echoMin > threshold / 100 * np.percentile(echoMin, 99)
        else:
            if len(signal.shape) > 3:
                echoMIP = np.sqrt(np.sum(np.abs(signal) ** 2, axis=3))
            else:
                echoMIP = np.sqrt(np.abs(signal) ** 2)
            tissueMask = echoMIP > threshold / 100 * np.max(echoMIP)

        if single_object:
            tissueMask = self._correct_for_single_object(tissueMask)
        return tissueMask

    def get_tissueMaskFilled(self, threshold=None, min_echo=False,
                             single_object=False, wfi_image=False, iDyn=None):
        if threshold is None: threshold = self.Masks['airSignalThreshold_percent']

        tissueMask = self.get_tissueMask(threshold=threshold, min_echo=min_echo,
                                         single_object=single_object,
                                         wfi_image=wfi_image, iDyn=iDyn)

        filledMask = np.zeros_like(tissueMask)
        for ie in range(0, tissueMask.shape[-1]):
            filledMask[:, :, ie] = binary_fill_holes(tissueMask[:, :, ie])

        return filledMask
    
    def set_VARPROparams(self):
        vp = self.VARPROparams

        TE_s = self.ImDataParams['TE_s']
        dTE = np.diff(TE_s)
        if np.sum(np.abs(dTE - dTE[0])) < 1e-5:
            dt = TE_s[1] - TE_s[0]
            vp['period'] = np.abs(1 / dt) / self.ImDataParams['centerFreq_Hz'] * 1e6
            print('Uniform TE spacing: Period = TE(2) - TE(1)')

        if 'FatModel' in self.AlgoParams:
            vp['signal_model'] = 'WF'
            assert self.ImDataParams['signal'].shape[-1] > 2
        else:
            vp['signal_model'] = 'W'
            assert self.ImDataParams['signal'].shape[-1] > 1

        vp['sampling_stepsize_fm'] = 2.0
        vp['sampling_stepsize_r2s'] = 2.0
        vp['range_r2s'] = np.array([0.0, 500.0])
        if 'period' in vp:
            period_ppm = vp['period']
            if period_ppm < 5:
                vp['range_fm_ppm'] = np.array([-2.5, 2.5], dtype=np.float32)
            else:
                vp['range_fm_ppm'] = np.array([-period_ppm, period_ppm], dtype=np.float32)
        else:
            vp['range_fm_ppm'] = np.array([-2.5, 2.5], dtype=np.float32)
        self.get_range_fm_Hz()

        try:
            precessionIsClockwise = self.ImDataParams['precessionIsClockwise']
        except:  # FIXME: bare except is bad practice
            precessionIsClockwise = 1
        if precessionIsClockwise <= 0:
            self.ImDataParams['signal'] = np.conj(self.ImDataParams['signal'])
            self.ImDataParams['precessionIsClockwise'] = 1

        if 'FatModel' in self.AlgoParams:
            vp['signal_model'] = 'WF'
        else:
            vp['signal_model'] = 'W'

    def get_range_fm_Hz(self):
        self.VARPROparams['range_fm'] = np.array(self.VARPROparams['range_fm_ppm']) * \
                                        self.ImDataParams['centerFreq_Hz'] * 1e-6
    
    def run_fieldmapping(self, ind_dynamic=None, init_fieldmap=None):
        if len(self.ImDataParams['signal'].shape) == 4:
            hmrGC = self.get_hmrGC_obj()
            hmrGC.verbose = True

            if init_fieldmap is None:
                method = 'multi-res'
            else:
                method = 'init'
                hmrGC.phasormap = init_fieldmap

            hmrGC.perform(method)

            self.WFIparams['method'] = f'{method}'
            self.WFIparams['fieldmap_Hz'] = hmrGC.fieldmap
            self.Masks['tissueMask'] = hmrGC.mask
            self.WFIparams['R2s_Hz'] = hmrGC.r2starmap
            self.set_T2s_ms()
            for key in hmrGC.images.keys():
                self.WFIparams[key] = hmrGC.images[key]
        else:
            self.WFIparams = {}
            nDyn = self.ImDataParams['signal'].shape[3]
            if ind_dynamic is None:
                range_dynamic = range(nDyn)
            else:
                range_dynamic = ind_dynamic
            for i in range_dynamic:
                print(f'Field-mapping for dyn {i+1}')
                tmp_obj = copy.deepcopy(self)
                tmp_obj.ImDataParams['signal'] = tmp_obj.ImDataParams['signal'][:, :, :, i, :]
                if 'tissueMask' in tmp_obj.Masks:
                    if len(tmp_obj.Masks['tissueMask'].shape) == 4:
                        tmp_obj.Masks['tissueMask'] = tmp_obj.Masks['tissueMask'][:, :, :, i]
                    else:
                        tmp_obj.Masks['tissueMask'] = tmp_obj.Masks['tissueMask']
                if i == 0:
                    init_fieldmap = None
                else:
                    init_fieldmap = self.WFIparams["fieldmap_Hz"][..., 0]*2*np.pi*np.diff(self.ImDataParams["TE_s"])[0]
                tmp_obj.run_fieldmapping(init_fieldmap=init_fieldmap)
                for key in tmp_obj.WFIparams:
                    arr = tmp_obj.WFIparams[key]
                    if isinstance(arr, np.ndarray):
                        shape = list(arr.shape)
                        shape.append(nDyn)
                        if key not in self.WFIparams.keys() or \
                           list(self.WFIparams[key].shape) != shape:
                            self.WFIparams[key] = np.zeros(shape, dtype=arr.dtype)
                        self.WFIparams[key][..., i] = arr
            method = tmp_obj.WFIparams['method']
            self.WFIparams['method'] = method
    
    def get_hmrGC_obj(self):
        if 'tissueMask' in self.Masks:
            mask = self.Masks['tissueMask']
        else:
            mask = self.get_tissueMask()
        signal = self.ImDataParams['signal']
        params = {}
        params['TE_s'] = self.ImDataParams['TE_s']
        params['voxelSize_mm'] = self.ImDataParams['voxelSize_mm']
        params['fieldStrength_T'] = self.ImDataParams['fieldStrength_T']
        params['centerFreq_Hz'] = self.ImDataParams['centerFreq_Hz']
        if 'FatModel' in self.AlgoParams:
            params['FatModel'] = self.AlgoParams['FatModel']
        if 'period' in self.VARPROparams:
            params['period'] = self.VARPROparams['period']
        if 'signal_model' in self.VARPROparams:
            params['signal_model'] = self.VARPROparams['signal_model']
        
        gandalf = MultiEcho(signal, mask, params)
        gandalf.sampling_stepsize_fm = self.VARPROparams['sampling_stepsize_fm']
        gandalf.sampling_stepsize_r2s = self.VARPROparams['sampling_stepsize_r2s']
        gandalf.range_fm_ppm = self.VARPROparams['range_fm_ppm']
        gandalf.range_r2s = self.VARPROparams['range_r2s']
        return gandalf

    def _save_params(self, params2save, savename=None, removelist=None, mat_file=True):
        save_params = getattr(self, params2save).copy()

        if mat_file:
            end = '.mat'
        else:
            end = '.pickle'
        path, filename = self._get_path_to_save(savename)
        if 'method' in save_params:
            savename = path + '/' + filename + '_' + params2save + '_' + \
                       save_params['method'] + end
        else:
            savename = path + '/' + filename + '_' + params2save + end

        if removelist:
            save_params = removeElementsInDict(save_params, removelist)

        print('save ' + savename, '...', end='')
        if mat_file:
            hdf5storage.savemat(savename, {params2save: save_params})
        else:
            with open(savename, 'wb') as f:
                pickle.dump(save_params, f)
        print('done!')

    def _get_path_to_save(self, savename):
        """Get the path and filename based on the passed savename

        :param savename: filename/path for saved object
        :returns: path, filename

        """
        if not savename:
            path = os.path.dirname(self.ImDataParams['filename'])
            if path == '':
                path = '.'
            filename = self.ImDataParams['fileID']
        else:
            if os.path.isdir(savename):
                path = os.path.dirname(savename)
                filename = self.ImDataParams['fileID']
            else:
                path = os.path.dirname(savename)
                filename = os.path.basename(savename)
        return path, filename
    
    def set_T2s_ms(self):
        if 'tissueMask' in self.Masks:
            mask = self.Masks['tissueMask']
        else:
            mask = self.get_tissueMask()

        if 'R2s_Hz' in self.WFIparams or 'waterR2s_Hz' in self.WFIparams or \
                'fatR2s_Hz' in self.WFIparams:
            self.WFIparams = get_T2s_ms(self.WFIparams, mask)

    def set_fatFraction_percent(self):
        ff = calculate_pdff_percent(self.WFIparams['water'],
                                    self.WFIparams['fat'])
        ff[np.isnan(ff)] = 0
        self.WFIparams['fatFraction_percent'] = ff



def recursively_load_attrs(h5file, path='/', load_data=False):
    """
    recursively load attributes for all groups and datasets in
    hdf5 file as python dict
    :param h5file: h5py.File(<filename>, 'r')
    :param path: "directory path" in h5 File
    :returns:
    :rtype: nested dicts
    """

    attrs_dict = {}
    for k, v in h5file[path].items():

        d = {}
        for ak, av in v.attrs.items():
            d[ak] = av

        if isinstance(v, h5._hl.dataset.Dataset):  # FIXME: call to a protected class function
            if load_data:
                attrs_dict[k] = np.array(v)
            else:
                attrs_dict[k] = d

        elif isinstance(v, h5._hl.group.Group):  # FIXME: call to a protected class function
            d.update(recursively_load_attrs(
                h5file, os.path.join(path, k), load_data))
            attrs_dict[k] = d

    return attrs_dict


def recursively_load_data(h5file, attrs_dict, path='/'):
    """
    recursively load data for all groups and datasets in
    hdf5 file as python dict corresponding to attrs_dict
    (see function recursively_load_attrs)
    :param h5file: h5py.File(<filename>, 'r')
    :param attrs_dict: output of function recursively_load_attrs
    :returns:
    :rtype: nested dicts
    """

    result = {}
    for k, v in attrs_dict.items():

        if k == '#refs#':
            continue

        if k == '#subsystem#':
            continue

        if isinstance(v, dict):

            if v.get('MATLAB_class') == b'function_handle':
                continue
            elif v.get('MATLAB_class') != b'struct':

                val = h5file[path + k + '/'][...]
                arrays3d = np.array(['signal', 'refsignal', 'fieldmap_Hz', 'R2s_Hz',
                                     'water', 'fat', 'silicone', 'fatFraction_percent'])
                if ~np.isin(k, arrays3d):
                    val = np.squeeze(val)

                if isinstance(val, np.ndarray) and \
                        val.dtype == [('real', '<f4'), ('imag', '<f4')]:
                    val = np.transpose(val.view(np.complex64)).astype(np.complex64)
                elif isinstance(val, np.ndarray) and \
                        (val.dtype == [('real', '<f8'), ('imag', '<f8')]):
                    val = np.transpose(val.view(np.complex)).astype(np.complex64)
                elif isinstance(val, np.ndarray) and \
                        (val.dtype == 'float64' or val.dtype == 'float32'):
                    val = (np.transpose(val).astype(np.float32))
                elif isinstance(val, np.ndarray) and \
                        (val.dtype == 'uint64' or val.dtype == 'uint32'):
                    val = (np.transpose(val).astype(np.uint32))
                elif isinstance(val, np.ndarray) and \
                        (val.dtype == 'bool_' or val.dtype == 'uint8'):
                    val = (np.transpose(val).astype(np.bool_))

                if v.get('MATLAB_class') == b'char':
                    try:
                        val = ''.join([chr(c) for c in val])
                    except:  # FIXME: bare except is bad practice
                        val = ''

                result[path + k + '/'] = val
            else:
                result.update(recursively_load_data(h5file, v, path + k + '/'))

    return result


def nest_dict(flat_dict):
    seperator = '/'
    nested_dict = {}
    for k, v in flat_dict.items():

        path_list = list(filter(None, k.split(seperator)))  # removes '' elements
        split_key = path_list.pop(0)
        left_key = seperator.join(path_list)

        if left_key == '':
            nested_dict[split_key] = v
            continue

        if not nested_dict.get(split_key):  # init new dict
            nested_dict[split_key] = {}

        if left_key != '':
            nested_dict[split_key].update({left_key: v})

    return nested_dict


def removeElementsInDict(inDict, listofstrings):
    for item in listofstrings:
        try:
            del inDict[item]
        except:  # FIXME: bare except is bad practice
            print('Dictionary has no item {}'.format(item))
    return inDict


def get_T2s_ms(inParams, mask):
    # mask = inParams['fieldmap_Hz'] != 0 # implicit mask definition
    if 'R2s_Hz' in inParams:
        T2s_ms = 1e3 / inParams['R2s_Hz']
        T2max = np.max(T2s_ms[~np.isinf(T2s_ms)])
        T2s_ms[np.isinf(T2s_ms)] = T2max
        inParams['T2s_ms'] = T2s_ms * mask

    if 'waterR2s_Hz' in inParams:
        wT2s_ms = 1e3 / inParams['waterR2s_Hz']
        T2max = np.max(wT2s_ms[~np.isinf(wT2s_ms)])
        wT2s_ms[np.isinf(wT2s_ms)] = T2max
        inParams['waterT2s_ms'] = wT2s_ms * mask

    if 'fatR2s_Hz' in inParams:
        fT2s_ms = 1e3 / inParams['fatR2s_Hz']
        T2max = np.max(fT2s_ms[~np.isinf(fT2s_ms)])
        fT2s_ms[np.isinf(fT2s_ms)] = T2max
        inParams['fatT2s_ms'] = fT2s_ms * mask

    return inParams

def plot_images(arr, cmap, planes, voxelSize_mm, position_3d, limits, filename='',
                fig_name=0, plot_cmap=False, patch=None, location_cmap="bottom", trim=True, base_arr=None):
    val_patch = []
    for plane in planes:
        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(frameon=False)
        #fig.set_size_inches(10,10)
        ax1 = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax1)
        
        values = copy.deepcopy(arr)
        if base_arr is not None:
            print("test")
            values_base = copy.deepcopy(base_arr)
        
        voxelSize_mm = voxelSize_mm
        cor_aspect = voxelSize_mm[2]/voxelSize_mm[0]
        sag_aspect = voxelSize_mm[2]/voxelSize_mm[1]
        trans_aspect = voxelSize_mm[1]/voxelSize_mm[0]
        if plane == 'coronal':
            values = np.transpose(values, [0, 2, 1])
            values = np.flip(values, axis=[1])
            if base_arr is not None:
                values_base = np.transpose(values_base, [0, 2, 1])
                values_base = np.flip(values_base, axis=[1])
            aspect = cor_aspect
            position = arr.shape[0]-position_3d[2]
        elif plane == 'sagittal':
            values = np.transpose(values, [1, 2, 0])
            values = np.flip(values, axis=1)
            if base_arr is not None:
                values_base = np.transpose(values_base, [1, 2, 0])
                values_base = np.flip(values_base, axis=1)
            aspect = sag_aspect
            position = position_3d[1]
        elif plane == 'axial':
            values = np.transpose(values, [2, 0, 1])
            if base_arr is not None:
                values_base = np.transpose(values_base, [2, 0, 1])
            aspect = trans_aspect
            position = position_3d[0]

        # import pdb; pdb.set_trace() 
        if trim:
            values, _ = trim_zeros(values[position])
            values = np.squeeze(values)
            if base_arr is not None:
                values_base, _ = trim_zeros(values_base[position])
                values_base = np.squeeze(values_base)
        else:
            values = np.squeeze(values[position])
            if base_arr is not None:
                values_base = np.squeeze(values_base[position])

        if limits is None:
            limits = [0,  np.percentile(values, 99)]

        if base_arr is not None:
            limits_base = [0,  np.percentile(values_base, 99)]
            ax1.imshow(values_base, vmin=limits_base[0], vmax=limits_base[1],
                        cmap='gray', aspect=aspect)
        im1 = ax1.imshow(values, vmin=limits[0], vmax=limits[1],
                         cmap=cmap, aspect=aspect)
        if patch:
            for i in range(len(patch)):
                x_coord = patch[i][0][0]
                x_size = patch[i][1]
                y_coord = patch[i][0][1]
                y_size = patch[i][2]
                values_rect = values[y_coord:y_coord+y_size, x_coord:x_coord+x_size]
                val_patch.append((np.mean(values_rect), np.std(values_rect)))
                rect = Rectangle(patch[i][0],x_size,y_size,linewidth=2,edgecolor='r',facecolor='none')
                ax1.add_patch(rect)
        # ax1.get_xaxis().set_visible(False)
        # ax1.get_yaxis().set_visible(False)
        if plot_cmap:
            # axins = inset_axes(ax, width = "5%", height = "100%", loc = 'lower left',
            #                    bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax1.transAxes,
            #                    borderpad = 0)
            # plt.colorbar(im1, cax = axins)
            if location_cmap == "right":
                im_ratio = values.shape[0]/values.shape[1]
            else:
                im_ratio = values.shape[1]/values.shape[0]
            plt.colorbar(im1, ax=ax1, fraction=0.046*im_ratio, pad=0.04, location=location_cmap)
            #plt.colorbar(im1, ax=ax1, location=location_cmap)
        plt.gca().set_axis_off()

        # Check if the directory exists
        if not os.path.exists(f'../plots/{fig_name}'):
            # Create the directory
            os.makedirs(f'../plots/{fig_name}')

        # Save the figure
        plt.savefig(f'../plots/{fig_name}/{filename}_{plane}.png', bbox_inches='tight', pad_inches=0, format="png")
        plt.close()
        #plt.show()
    return 

def trim_zeros(arr, margin=0, apply_to=None):
    '''
    Trim the leading and trailing zeros from a N-D array.
    :param arr: numpy array
    :param margin: how many zeros to leave as a margin
    :returns: trimmed array
    :returns: slice object
    Christof Boehm,
    christof.boehm@tum.de
    '''
    s = []
    for dim in range(arr.ndim):
        start = 0
        end = -1
        slice_ = [slice(None)]*arr.ndim

        go = True
        while go:
            slice_[dim] = start
            go = not np.any(arr[tuple(slice_)])
            start += 1
        start = max(start-1-margin, 0)

        go = True
        while go:
            slice_[dim] = end
            go = not np.any(arr[tuple(slice_)])
            end -= 1
        end = arr.shape[dim] + min(-1, end+1+margin) + 1

        s.append(slice(start,end))
    if apply_to is None:
        return arr[tuple(s)], tuple(s)
    else:
        return apply_to[tuple(s)]#, tuple(s)