"""Pure helpers for request-correlated heavy perception refreshes."""


def stamp_dict_to_ns(value):
    if not isinstance(value, dict):
        raise ValueError('timestamp must be an object')
    try:
        sec = int(value.get('sec', 0))
        nanosec = int(value.get('nanosec', 0))
    except (TypeError, ValueError):
        raise ValueError('timestamp fields must be integers')
    if sec < 0 or nanosec < 0 or nanosec >= 1000000000:
        raise ValueError('timestamp fields are outside their valid range')
    return sec * 1000000000 + nanosec


def ros_stamp_to_dict(stamp):
    return {'sec': int(stamp.sec), 'nanosec': int(stamp.nanosec)}


def request_minimum_stamp_ns(request):
    value = request.get('min_image_stamp')
    if value is None:
        return None
    stamp_ns = stamp_dict_to_ns(value)
    if stamp_ns <= 0:
        raise ValueError('min_image_stamp must be nonzero')
    return stamp_ns


def image_satisfies_request(request, image_stamp):
    minimum = request_minimum_stamp_ns(request)
    if minimum is None:
        return True
    return stamp_dict_to_ns(ros_stamp_to_dict(image_stamp)) >= minimum
