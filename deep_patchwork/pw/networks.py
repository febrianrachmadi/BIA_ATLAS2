
import torch
from . import tools as tls
from .units import rebrand,UNet_factory,FFT_Net_factory,DummyNet


def UnetADD(params,verbose=0):
    U_net = UNet_factory(**params)

    if params["fin"][1] == 0:
        return U_net

    net = rebrand(U_net["net"],"_unet")    
    
    net["IBN"] =  {"op":"nn_op",
              "params":{"op":"BatchNorm","num_features": params["fin"][1]},
              "in":["IN1"],
              "args":"none"}
  
    net["enc0_unet"]["in"]=["IN0","IBN"]


    net['ISEL'] = {"op":"lambda","params":"lambda x:x.clone().detach()",
                            "range":[0,1],
                            "in": ["IN1"],
                            }
    net["OUT"] =  {"op":"basic",
                  "params":{"btype":"add"},
                  "in":["out_unet",'ISEL'],
                  "args":"none"}
    new_net = {}
    new_net["params"] = {}
    new_net["params"]["fin"] = params["fin"]
    new_net["params"]["fout"] = params["fout"]
    new_net["params"]["dim"] = params["dim"]
    new_net["params"]["pfield"] = params["pfield"]
    new_net["params"]["name"] = params["name"]
    new_net["net"] = net
    
    return new_net

def RegNet_simple_ND(params,verbose=0):
    
    unit_types = {"fnet":[0,1]}
    if "unit_types" in params:
        unit_types = params["unit_types"]
    
    use_fft = (params["scale"][1]-params["scale"][0]) in unit_types["fnet"] 
    
    params["fft_net"]["pfield"] = params["pfield"]
    params["fft_net"]["fin"] = params["fin"]
    params["fft_net"]["fout"] = params["fout"]
    
    params["coord_net"]["pfield"] = params["pfield"]
    params["coord_net"]["fin"] = params["fin"]
    params["coord_net"]["fout"] = 6#params["fout"]
    #params["coord_net"]["fout"] = params["fout"]
    
    
    if use_fft:
        print("FT-Net")
        fft_net = FFT_Net_factory(**params["fft_net"])
    else:
        print("Coord-Net")
        fft_net = UNet_factory(**params["coord_net"])
    
    return fft_net

def RegNet_simple_ND_add2(params,verbose=0):
    
    unit_types = {"fnet":[0,1]}
    if "unit_types" in params:
        unit_types = params["unit_types"]
    
    use_fft = (params["scale"][1]-params["scale"][0]) in unit_types["fnet"] 
    
    params["fft_net"]["pfield"] = params["pfield"]
    params["fft_net"]["fin"] = params["fin"]
    params["fft_net"]["fout"] = params["fout"]
    
    params["coord_net"]["pfield"] = params["pfield"]
    params["coord_net"]["fin"] = params["fin"]
    params["coord_net"]["fout"] = 6#params["fout"]
    #params["coord_net"]["fout"] = params["fout"]
    
    
    if use_fft:
        print("FT-Net")
        fft_net = FFT_Net_factory(**params["fft_net"])
    else:
        print("Coord-Net")
        fft_net = UNet_factory(**params["coord_net"])


    if params["fin"][1] == 0:
        return fft_net
    
    net = rebrand(fft_net["net"],"_fft")    
    net["enc0_fft"]["in"]=["IN0","IN1"]
   
    
    net['ISEL'] = {"op":"lambda","params":"lambda x:x.clone().detach()",
                            "in": ["IN1"],
                            }
    
    net["OUT"] =  {"op":"basic",
                  "params":{"btype":"add"},
                  "in":["out_fft","ISEL"],
                  "args":"none"}
    
  
    new_net = {}
    new_net["params"] = {}
    new_net["params"]["fin"] = params["fin"]
    new_net["params"]["fout"] = params["fout"]
    new_net["params"]["dim"] = params["dim"]
    new_net["params"]["pfield"] = params["pfield"]
    new_net["params"]["name"] = params["name"]
    new_net["net"] = net
    
    return new_net


def RegNet_simple_ND_add(params,verbose=0):
    
    unit_types = {"fnet":[0,1]}
    if "unit_types" in params:
        unit_types = params["unit_types"]
    
    use_fft = (params["scale"][1]-params["scale"][0]) in unit_types["fnet"] 
    
    params["fft_net"]["pfield"] = params["pfield"]
    params["fft_net"]["fin"] = params["fin"]
    params["fft_net"]["fout"] = params["fout"]
    
    params["coord_net"]["pfield"] = params["pfield"]
    params["coord_net"]["fin"] = params["fin"]
    params["coord_net"]["fout"] = 6#params["fout"]
    #params["coord_net"]["fout"] = params["fout"]
    
    
    if use_fft:
        print("FT-Net")
        fft_net = FFT_Net_factory(**params["fft_net"])
    else:
        print("Coord-Net")
        fft_net = UNet_factory(**params["coord_net"])


    if params["fin"][1] == 0:
        return fft_net
    
    net = rebrand(fft_net["net"],"_fft")    
    net["enc0_fft"]["in"]=["IN0","IN1"]
   
    net["OUT"] =  {"op":"basic",
                  "params":{"btype":"add"},
                  "in":["out_fft","IN1"],#"in":["out_fft","I1_valid"],
                  "args":"none"}
    
  
    new_net = {}
    new_net["params"] = {}
    new_net["params"]["fin"] = params["fin"]
    new_net["params"]["fout"] = params["fout"]
    new_net["params"]["dim"] = params["dim"]
    new_net["params"]["pfield"] = params["pfield"]
    new_net["params"]["name"] = params["name"]
    new_net["net"] = net
    
    return new_net





def RegNet_builder_ND(params,verbose=0):
    #def __init__(self,params,verbose=0):
            #super().__init__()
            
            print("###################",params["scale"],"############################")
            FC = params["FC"]
            sample_with_offsets = False
            if "grid_offsets" in params:
                sample_with_offsets = params["grid_offsets"]
            #patchlevels = 4
            #if "patchlevels" in params:
            #    patchlevels = params["patchlevels"]
            unit_types = {"fnet":[0,1]}
            if "unit_types" in params:
                unit_types = params["unit_types"]
            fft_net_with_ref = False
            if "fft_net_with_ref" in params:
                fft_net_with_ref = params["fft_net_with_ref"]
            
            use_img2img_as_input = True
            if "use_img2img_as_input" in params and params["use_img2img_as_input"]:
                use_img2img_as_input = True
            use_img_as_input = False
            if "use_img_as_input" in params and params["use_img_as_input"]:
                use_img_as_input = True

            assert(use_img_as_input or use_img2img_as_input)
            fti = 1 if use_img2img_as_input and use_img_as_input else 0
            #fto = 1 if use_img_as_input else 0
            


            print("FC: ",FC)
            dim = params["dim"]
            
            
            #verbose = 2
            params["img_net"]["fin"] = params["fin"][0]
            params["img_net"]["pfield"] = params["pfield"]
            
            if params["fin"][1] > 0:
                #params["img_net"]["fout"] -= 1
                params["img_net"]["fin"] += 1
                #if sample_with_offsets:
                    #params["img_net"]["fin"] += 6
                
                    
                
                
            #print("im_net fout: ",params["img_net"]["fout"])
            
            img_net = UNet_factory(**params["img_net"])
            
            if params["fin"][1] > 0:
                #params["fft_net"]["fin"] = [params["fin"][0],5]
                if FC:
                   fft_net_in_prop = dim*2 #+  params["cc_net"]["features"] * 2 + 1 
                else:
                   fft_net_in_prop = dim #+  params["cc_net"]["features"] * 2 + 1
                params["fft_net"]["fin"] = [params["fin"][0]+fti,fft_net_in_prop]
                
                
                #print("use_img2img_as_input ",fti)
                if sample_with_offsets:
                    params["coord_net"]["fin"] = [params["fin"][0]+7+fti,fft_net_in_prop]
                    if fft_net_with_ref:
                        params["fft_net"]["fin"] = [params["fin"][0]+7,fft_net_in_prop]
                else:
                    params["coord_net"]["fin"] = [params["fin"][0]+1+fti,fft_net_in_prop]
                    if fft_net_with_ref:
                        params["fft_net"]["fin"] = [params["fin"][0]+1,fft_net_in_prop]
                
            else:
                params["fft_net"]["fin"] = [params["fin"][0]+fti,params["fin"][1]]
                params["coord_net"]["fin"] = [params["fin"][0]+fti,params["fin"][1]]
            print("params wtf",params["fft_net"]["fin"])
            print("params wtf",params["coord_net"]["fin"])
                #params["fft_net"]["fin"][0] += fti
                #params["coord_net"]["fin"][0] += fti
                
            
            params["fft_net"]["pfield"] = params["pfield"]
            params["coord_net"]["pfield"] = params["pfield"]
            
#            use_fft = params["scale"][0]>1
            #use_fft = params["scale"][0]>patchlevels-3
            use_fft = (params["scale"][1]-params["scale"][0]) in unit_types["fnet"] 
            
            if use_fft:
                print("FT-Net")
                fft_net = FFT_Net_factory(**params["fft_net"])
            else:
                print("Coord-Net")
                fft_net = UNet_factory(**params["coord_net"])
            
            #print(["net"].keys())
            img_net["net"] = rebrand(img_net["net"],"_img")
            fft_net["net"] = rebrand(fft_net["net"],"_fft")
            net = {**img_net["net"], **fft_net["net"]}
            

            i2i_out = "out_img"
            i2i_out = "out_img_sig"
            net["out_img_sig"] = {"op":"Sigmoid","params":{},
                                          "in": ["out_img"],
                                         }

            if params["fin"][1] == 0:
                IN0 = []
                if use_img_as_input:
                    IN0 += ["IN0"] 
                if use_img2img_as_input:
                    #IN0 = "out_img"
                    print("using translated image as input for the coordinate prediction (detached)")
                    #net["Timg"] = {"op":lambda x:x.clone().detach(),"params":{},#{"op":"Identity","params":{},
                    net["Timg"] = {"op":"lambda","params":"lambda x:x.clone().detach()",#{"op":"Identity","params":{},
                                                                 "in": [i2i_out],
                                                                 }
                    IN0 += ["Timg"]

                net['enc0_img']["in"] = ["IN0"]
                net['enc0_fft']["in"] = IN0
                
                net["OUT"] = {"op":"Identity","params":{},
                                          "in": ["out_fft",i2i_out],
                                         }
            else:
                #net['img_net_in_prop'] =  {"op":lambda x:x.clone().detach(),"params":{},
                net['img_net_in_prop'] =  {"op":"lambda","params":"lambda x:x.clone().detach()",
                                                             "range":[-2,-1],
                                                             "in": ["IN1"],
                                                             }
                
                net['enc0_img']["in"] = ["IN0","img_net_in_prop"]
              
                if FC:
                    net['I1_valid'] = {"op":"Identity",
                                                                  "params":{},
                                                                 "range":[0,2*dim],#"range":[0,5],
                                                                 "in": ["IN1"],
                                                                 }
                    
                    if not use_fft or fft_net_with_ref:
                    #if True:
                        
                        if sample_with_offsets:
                            offsets = []
                            #for x in [-1,1]:
                            #    for y in [-1,1]:
                            #        for z in [-1,1]:
                            #            o = torch.tensor([x,y,z])[None,None,None,None,:]
                            #            o *= (params["scale"][0]+1)
                            #            offsets += [o]
                            sc = (params["scale"][0]+1)
                            offsets += [sc*torch.tensor([-1,0,0])[None,None,None,None,:]]
                            offsets += [sc*torch.tensor([1,0,0])[None,None,None,None,:]]
                            offsets += [sc*torch.tensor([0,-1,0])[None,None,None,None,:]]
                            offsets += [sc*torch.tensor([0,1,0])[None,None,None,None,:]]
                            offsets += [sc*torch.tensor([0,0,-1])[None,None,None,None,:]]
                            offsets += [sc*torch.tensor([0,0,1])[None,None,None,None,:]]
                            offsets += [sc*torch.tensor([0,0,0])[None,None,None,None,:]]
                            
                                        #params["scale"][0]
                            net['warp'] = {"op":"grid_sample_layer","params":{"offsets":offsets },
                                                      "in": ["I1_valid"],
                                                 }
                        else:
                            net['warp'] = {"op":"grid_sample_layer","params":{},
                                                      "in": ["I1_valid"],
                                                 }

                            
                #else:
                #    net['I1_valid'] = {"op":lambda x:x.clone().detach(),#"op":"Identity",
                #                        "params":{},
                #                          "range":[0,dim],#"range":[0,3],
                #                          "in": ["IN1"],
                #                          }
                #    if not use_fft:
                #        net['warp'] = {"op":"grid_sample_layer","params":{"CT":"CAR"},
                #                                  "in": ["I1_valid"],
                #                                 }
                
                    
                IN0 = []
                if use_img_as_input:
                    IN0 += ["IN0"]    
                if use_img2img_as_input:
                    #IN0 = "out_img"
                    print("using translated image as input for the coordinate prediction (detached)")
                    
                    #net["Timg"] = {"op":lambda x:x.clone().detach(),"params":{},
                    net["Timg"] = {"op":"lambda","params":"lambda x:x.clone().detach()",                                 
                                                                 "in": [i2i_out],
                                                                 }
                    IN0 += ["Timg"]

                #if use_fft or fft_net_with_ref:
                if use_fft and not fft_net_with_ref:
                    #net['enc0_fft']["in"] = ["IN0","I1_valid"]
                    net['enc0_fft']["in"] = IN0 + ["I1_valid"]
                else:
                #if True:
                    print("using warped template")
                    #net['enc0_fft']["in"] = ["IN0","warp","I1_valid"]
                    net['enc0_fft']["in"] = IN0 + ["warp","I1_valid"]
                    
                net["fft_update"] =  {"op":"basic",
                              "params":{"btype":"add"},
                              "in":["out_fft","I1_valid"],
                              "args":"none"}
                
                net["OUT"] = {"op":"Identity","params":{},
                                          "in": ["fft_update",i2i_out],
                                         }
        
           
           # print(params)
            new_net = {}
            new_net["params"] = {}
            new_net["params"]["fin"] = params["fin"]
            new_net["params"]["fout"] = params["fout"]
            new_net["params"]["dim"] = params["dim"]
            new_net["params"]["pfield"] = params["pfield"]
            new_net["params"]["name"] = params["name"]
            
            new_net["net"] = net
            #self.mymodel = model_factory(new_net,verbose=verbose)
            return new_net


def DummyNet_builder(params,verbose=1):
        mynet = DummyNet(**params)
        return mynet
        

def unet_model_builder(params,verbose=1):
        return UNet_factory(**params)
        
        

def fft_prop_test_model_builder(params,verbose=0):
    #def __init__(self,params,verbose=0):
        #super().__init__()
        if params["scale"][0]==params["scale"][1]:
            mynet = FFT_Net_factory(**params)
        else:
            params_ = {}
            params_["dim"] = params["dim"]
            mynet = DummyNet(**params_)
            mynet["net"]['OUT'] =  {"op":"Identity","params":{},
                                                 "in": ["IN1"],
                                                }
            
            return mynet