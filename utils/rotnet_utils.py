import torch


def rotate_batch_with_labels(batch, labels):
    pts_list = []
    for pts, label in zip(batch, labels):
        if label == 1:
            pts = rotate_point_cloud_by_angle(pts, rotation_angle=90)
        elif label == 2:
            pts = rotate_point_cloud_by_angle(pts, rotation_angle=180)
        elif label == 3:
            pts = rotate_point_cloud_by_angle(pts, rotation_angle=270)
        pts_list.append(pts.unsqueeze(0))
    return torch.cat(pts_list)


def rotate_batch(batch, label='rand'):
    if label == 'rand':
        labels = torch.randint(4, (len(batch),), dtype=torch.long)
    elif label == 'expand':
        labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
                            torch.zeros(len(batch), dtype=torch.long) + 1,
                            torch.zeros(len(batch), dtype=torch.long) + 2,
                            torch.zeros(len(batch), dtype=torch.long) + 3])
        batch = batch.repeat((4, 1, 1, 1))
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long) + label

    return rotate_batch_with_labels(batch, labels), labels


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          1xNx3 array, original batch of point clouds
        Return:
          1xNx3 array, rotated batch of point clouds
    """
    rotation_angle = torch.tensor(rotation_angle).cuda()
    rotated_data = torch.zeros(batch_data.shape).cuda()
    # for k in range(batch_data.shape[0]):
    cosval = torch.cos(rotation_angle).cuda()
    sinval = torch.sin(rotation_angle).cuda()
    rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]]).cuda()
    shape_pc = batch_data[:, 0:3]
    rotated_data[:, 0:3] = torch.matmul(shape_pc.reshape((-1, 3)), rotation_matrix.cuda())
    return rotated_data.cuda()