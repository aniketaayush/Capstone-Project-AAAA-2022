from core.data_provider import comedian

datasets_map = {
    'action': comedian,
}


def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width, seq_length, injection_action,train_end_no,test_end_no, use_start_no, use_end_no, input_length, run_scenario=1):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
    if dataset_name == 'action':
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'test_end_no':test_end_no,
                       'train_end_no':train_end_no,
                       'use_start_no':use_start_no,
                       'use_end_no':use_end_no,
                       'input_length':input_length,
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if run_scenario == 1:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        elif run_scenario == 2:
            use_input_handle = input_handle.get_use_input_handle()
            use_input_handle.begin(do_shuffle=False)
            return use_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle