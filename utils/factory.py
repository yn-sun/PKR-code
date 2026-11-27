def get_model(model_name, args):
    name = model_name.lower()
    if name == "pkr":
        from models.pkr import PKR
        return PKR(args)
    raise NotImplementedError("Unknown model {}".format(model_name))
