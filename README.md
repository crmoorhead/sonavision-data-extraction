# Image Extraction from Sonavision Data Files

The data files PDF details how the data is extracted from the the files generated by the Sonavision software recording the SV1010 and how those files were deciphered. This readme will give some documentation for the functions written in the code. 

__experiment_loader(dir_name)__: Takes an argument of the path of a directory containing all the scan data files. The output is a dictionary with the file names as keys and an array containing the corresponding raw data as values.

__file_settings(file)__: Takes an array associated with a file name keyword from the experiment loader dictionary and prints recording information to the screen. This includes and changes in the settings made withing the duration of the recording. This includes settings for scan speed, scan mode, scan frequency, resolution, range and other relevant details.

__print_stats(data)__: This produces an abridged version of the file_settings function above, with the same input. It makes the assumption that the settings were not changed during the recording session.

__data_slices(data,\*args)__: Returns the scan mode as the first output and an array of extracted data with irrelevant information discarded as the second output. The mode will be an integer (this is how it is stored n the original data) to be understood as follows: Stop/Sidescan (0), Sector Scan (1), Flyback (2), Rotate (3). The data slices array is a list of lists with each list starting with the angle at the current time step followed by the intensities of the responses from the area closest to the scanner up until the full range. The data portion will be of length 400 at all scan speeds except for Super Fast where it is 200. If the mode of scanning schanges during the recording, an error message will occur.

__preprocess_data_slices(data_slices,\*args)__: This takes the data slices array and performs minor preprocessing. Where the angle of two or more consequetive readings is the same, it will merge these together to create one single list of intensities for that angle. In the replay using the Sonavision software, the data is rewritten over the same position during the animation. It is the assumption that comining these readings will reduce noise in that portion of the scan. 

_NB: This cannot be applied to Stop mode as it would combine all entries to one single data slice as, by definitiion, the angle is constant. To extract images from this type of data, a different strategy is used._

__extract_angles(data_slices)__: A helper function that just returns a list of all the angles for each data slice. Used to check upper and lower limits etc in other functions.

__forb(a,b)__: This determines whether the scanner is moving forward of backwards. Note that the processed data slices should never have any consequetive slices recorded at the same angle if they are preprocessed using the function described earlier. The necessity for this function is centred around the fact that angles are stored in base 1024. A jump from 1022 to 4 is thus in the forward direction and if the scanner crosses the zero angle at any time, adjustments for the direction need to be made. A forward jump (clockwise) outputs 1 and a backward jump (anticlockwise) outputs -1. No change results in 0. By default, forward direction is the direction of the shortest path round the circle from a to b. 

__block_extractor(data_slices,mode,\*args,\*\*kwargs)__: This function will divide the scan up into blocks in a manner intuitive to the nature of the scan. For stop/side scan mode, it breaks up the scan into equal-sized chunks with a designated overlap defined a an intger number of pixels by an optional "overlap" keyword argument. For scan sector mode, the blocks are determined by the limits of the sector defined by the change in direction of the scan. For flyback, the blacks have the same limits but with new blocks determined by detecting a jump to the start of the sector. Finally, in rotation mode, the blocks are determined by the start position of the scan and a new block started after 360 degrees have passed. An optional keyword argument "parts" will split the scan into an integer number of sectors. This can be combined with the "overlap" keyword. The output is an list of blocks where each block is a list of lists.

__flip_merge(test ?)__: Even though the scanning procedure should be one-directional for the scan and flyback modes, the recorded values for the 

__dir_in_b(a,b,base)__:

__sort_mod(sorted_list,jump)__:

__sort_lists_mod(list,jump)__:

__flip_merge_data(test_data)__:

__flip_anticlockwise(data_blocks)__:

__get_limits(data_blocks)__:

__block_generator(blocks, depth, limits, sparity, jitter, empty)__:

__linear_interp(list1,list2,intervals)__:

__pad(data_blocks, limits, mode)__:

__image_format(full_data_blocks,\*args)__:

__images_from_data(file_array, \*args)__:

__process_directory(source_dir, target_dir)__:
