%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DEFINE_TRACERS: Return a list of tracers info to fill netcdf files %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
    tracers= {...
        {'N2O'        ,'Nitrous oxide'                                    ,'mMol N2O m-3'   }
        {'N2'         ,'diazote'                                          ,'mMol N2  m-3'   }
        {'NO2'        ,'Nitrite'                                          ,'mMol N   m-3'   }
        {'N2O_SIDEN'  ,'Nitrous oxide from boundary'                      ,'mMol N2O m-3'   }
        {'N2O_NEV'    ,'Nitrous oxide (Nevisson)'                         ,'mMol N2O m-3'   }
        {'N2O_ATM'    ,'Nitrous oxide from atmosphere'                    ,'mMol N2O m-3'   }
        {'PO4'        ,'Phosphate'                                        ,'mMol PO4 m-3'   }
        {'NO3'        ,'Nitrate'                                          ,'mMol No3 m-3'   }
        {'SiO3'       ,'Silicate'                                         ,'mMol SiO3 m-3'  }
        {'NH4'        ,'Ammonia'                                          ,'mMol PO4 m-3'   }
        {'Fe'         ,'Iron'                                             ,'mMol Fe m-3'    }
        {'O2'         ,'Oxygen'                                           ,'mMol O2 m-3'    }
        {'DIC_glodap' ,'Dissolved inorganic carbon from glodap'           ,'muMol kg-1'     }
        {'Alk_glodap' ,'Alkalinity from glodap'                           ,'muMol kg-1'     }
        {'DIC'        ,'Dissolved inorganic carbon'                       ,'mMol C m-3'     }
        {'Alk'        ,'Alkalinity'                                       ,'mmol m-3'       }
        {'DOC'        ,'Dissolved organic carbon'                         ,'mMol C m-3'     }
        {'SPC'        ,'Small phytoplankton carbon'                       ,'mMol C m-3'     } % i.e. 0.75*5*SPCHL (MF: initiate with lower Carbon!)
        {'SPCHL'      ,'Small phytoplankton chlorophyll'                  ,'mg Chla m-3'    } % for diaz_to_coocos: 0.45
        {'SPCACO3'    ,'Small phytoplankton carbonate'                    ,'mmol CaCO3 m-3' } % i.e. 0.1*SPCHL
        {'DIATC'      ,'Diatom carbon'                                    ,'mMol C m-3'     } % i.e. DIATC=0.75*3*DIATCHL=0.75*3*0.1*SPCHL (MF: initiate with lower Carbon!)
        {'DIATCHL'    ,'Diatom chlorophyll'                               ,'mg Chla m-3'    } % i.e. 0.1*SPCHL=0.1*0.9*CHL
        {'ZOOC'       ,'Zooplankton carbon'                               ,'mg C m-3'       } % i.e. ZOOC=0.4*SPC=0.4*5*SPCHL
        {'SPFE'       ,'Small phytoplankton iron'                         ,'mmol Fe m-3'    }
        {'DIATSI'     ,'Diatom silicate'                                  ,'mmol Si m-3 '   } % i.e. 0.1*DIATCHL
        {'DIATFE'     ,'Diatom iron'                                      ,'mmol Fe m-3'    } % i.e. 2e-5*DIATCHL
        {'DIAZC'      ,'Diazotroph carbon'                                ,'mmol C m-3'     } % for diaz_to_coocos: 5  (MF: initiate with lower Carbon!)
        {'DIAZCHL'    ,'Diazotroph chlorophyll'                           ,'mg Chla m-3'    } % i.e. DIAZCHL=0.01/0.9*SPCHL=0.01*CHL; for diaz_to_coocos: 1 (1*SPCHL)   
        {'DIAZFE'     ,'Diazotroph iron'                                  ,'mmol Fe m-3'    } % for diaz_to_coocos: 1e-4*0.01/0.9   
        {'DIC_anth'   ,'Anthropogenic DIC'                                ,'mMol C m-3'     } % AF
        {'DON'        ,'Dissolved organic nitrogen'                       ,'mMol N m-3'     }
        {'DONr'       ,'refractory Dissolved organic nitrogen'            ,'mMol N m-3'     }
        {'DONR'       ,'refractory Dissolved organic nitrogen'            ,'mMol N m-3'     }
        {'DOFe'       ,'Dissolved organic iron'                           ,'mMol Fe m-3'    }
        {'DOFE'       ,'Dissolved organic iron'                           ,'mMol Fe m-3'    }
        {'DOP'        ,'Dissolved organic phosphorus'                     ,'mMol P m-3'     } 
        {'DOPr'       ,'refractory Dissolved organic phosphorus'          ,'mMol P m-3'     } 
        {'DOPR'       ,'refractory Dissolved organic phosphorus'          ,'mMol P m-3'     }
        {'CHLA'       ,'Chlorophyll A'                                    ,'mg Chla m-3'    }
        {'DIC_GLODAP' ,'Dissolved inorganic carbon'                       ,'muMol kg-1'     } % AF
        {'Alk_GLODAP' ,'Alkalinity'                                       ,'muMol kg-1'     } % AF
        {'basindx'    ,'Basin Index'                                      ,'-'              }
        {'TA_srf'     ,'Surface  Total Alkalinity'                        ,'muMol kg-1'     } % AF
        {'NTA_srf'    ,'Surface Normalized Total Alkalinity'              ,'muMol kg-1'     } % AF
        {'DIC_srf'    ,'Surface dissolved inorganic carbon (pcO2 based)'  ,'muMol kg-1'     } % AF
        {'NPP_VGPM'   ,'Depth integrated NPP (SeaWiFS VGPM estimate)'     ,'mMol C m-2 s-1' } % convert 'mg C d-1' to 'mMol C s-1'
        {'NPP_CbPM'   ,'Depth integrated NPP (SeaWiFS CbPM estimate)'     ,'mMol C m-2 s-1' } % convert 'mg C d-1' to 'mMol C s-1'
        {'seawifs_chl',' SeaWiFS CHL estimate from ocean color'           ,'mg Chla m-3 '   } % convert 'mg C d-1' to 'mMol C s-1'
        {'PIC'        ,'Particulate Inorganic carbon'                     ,'mMol C'         }
        {'pCO2'       ,'Surface water pCO2'                               ,'uatm'           }
        };

for trc=1:length(tracers)    
    test = tracers{trc} ;
    bgctracers_list.name{trc}     = test{1} ;
    bgctracers_list.longname{trc} = test{2} ;
    bgctracers_list.units{trc}    = test{3} ;
end




