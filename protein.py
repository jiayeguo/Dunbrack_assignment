"""
protein_features.py
This is a tool to featurize kinase conformational changes through the entire Kinome.

"""

def key_klifs_residues(numbering):
    """
    Retrieve a list of PDB residue indices relevant to key kinase conformations mapped via KLIFS.

    Define indices of the residues relevant to a list of 12 collective variables relevant to
    kinase conformational changes. These variables include: angle between aC and aE helices,
    the key K-E salt bridge, DFG-Phe conformation (two distances), X-DFG-Phi, X-DFG-Psi,
    DFG-Asp-Phi, DFG-Asp-Psi, DFG-Phe-Phi, DFG-Phe-Psi, DFG-Phe-Chi1, and the FRET L-S distance.
    All features are under the current numbering of the structure provided.

    Parameters
    ----------
    numbering : list of int
        numbering[klifs_index] is the residue number for the given PDB file corresponding to KLIFS residue index 'klifs_index'

    Returns
    -------
    key_res : list of int
        Key residue indices

    """
    if numbering == None:
        print("The structure was not found in the klifs database.")
        key_res = None
        return key_res

    key_res = dict() #initialize key_res (which read from the 0-based numbering list)
    for i in range(5):
        key_res[f'group{i}'] = list()
    ## feature group 0: A-loop backbone dihedrals
    key_res['group0'].append(numbering[83]) # start of A-loop

    ## feature group 1: P-loop backbone dihedrals
    key_res['group1'].append(numbering[3]) # res0 in P-loop
    key_res['group1'].append(numbering[4]) # res1 in P-loop
    key_res['group1'].append(numbering[5]) # res2 in P-loop
    key_res['group1'].append(numbering[6]) # res3 in P-loop
    key_res['group1'].append(numbering[7]) # res4 in P-loop
    key_res['group1'].append(numbering[8]) # res5 in P-loop

    ## feature group 2: aC-related features
    #angle between aC and aE helices
    key_res['group2'].append(numbering[19])  # res0 in aC
    key_res['group2'].append(numbering[29])  # res10 in aC
    key_res['group2'].append(numbering[62])  # end of aE

    # key salt bridge
    key_res['group2'].append(numbering[16])  # K in beta III
    key_res['group2'].append(numbering[23])  # E in aC

    ## feature group 3: DFG-related features
    key_res['group3'].append(numbering[79])  # X-DFG
    key_res['group3'].append(numbering[80])  # DFG-Asp
    key_res['group3'].append(numbering[81])  # DFG-Phe
    key_res['group3'].append(numbering[27])  # ExxxX

    ## feature group 4: the FRET distance
    # not in the list of 85 (equivalent to Aura"S284"), use the 100% conserved beta III K as a reference
    key_res['group4'].append(numbering[16] + 120)

    # not in the list of 85 (equivalent to Aura"L225"), use the 100% conserved beta III K as a reference
    key_res['group4'].append(numbering[16] + 61)

    return key_res

def compute_simple_protein_features(traj, key_res):
    """
    This function takes the PDB code, chain id and certain coordinates of a kinase from
    a command line and returns its structural features.

    Parameters
    ----------
    traj : str
	A MDTraj.Trajectory object of the input structure (a pdb file or a simulation trajectory).
    key_res : dict of int
        A dictionary (with keys 'group0' ... 'group4') of feature-related residue indices in five feature groups.
    Returns
    -------
    features: list of floats
    	A list (single structure) or lists (multiple frames in a trajectory) of 72 features in 5 groups (A-loop, P-loop, aC, DFG, FRET)

    .. todo :: Use kwargs with sensible defaults instead of relying only on positional arguments.


    """
    import mdtraj as md
    import numpy as np
    import pandas as pd

    topology = traj.topology
    coord = traj.xyz

    # JG debug
    #table, bonds = topology.to_dataframe()
    #atoms = table.values
    #print(atoms)

    # get the array of atom indices for the calculation of:
    #       * 62 dihedrals (a 62*4 array where each row contains indices of the four atoms for each dihedral)
    #       * 7 ditances (a 7*2 array where each row contains indices of the two atoms for each distance)
    dih = np.zeros(shape=(62, 4), dtype=int, order='C')
    dis = np.zeros(shape=(7, 2), dtype=int, order='C')

    # name list of the features
    feature_names = list()

    # parse the topology info
    '''
    The coordinates are located by row number (usually is atom index minus one, which is also why it's zero-based)
    by mdtraj but when the atom indices are not continuous there is a problem so a safer way to locate the coordinates
    is through row number (as a fake atom index) in case the atom indices are not continuous.
    '''
    ### dihedrals
    count = 0
    ## feature group 0: A-loop backbone dihedrals (Phi, Psi)
    for i in range(key_res['group0'][0], key_res['group0'][0] + 21): #there are 21 residues in A-loop
        if topology.select(f"chainid 0 and residue {i - 1} and name C"):
            dih[count][0] = topology.select(f"chainid 0 and residue {i - 1} and name C")

        if topology.select(f"chainid 0 and residue {i} and name N"):
            dih[count][1] = topology.select(f"chainid 0 and residue {i} and name N")

        if topology.select(f"chainid 0 and residue {i} and name CA"):
            dih[count][2] = topology.select(f"chainid 0 and residue {i} and name CA")

        if topology.select(f"chainid 0 and residue {i} and name C"):
            dih[count][3] = topology.select(f"chainid 0 and residue {i} and name C")

        dih[count+1][0] = dih[count][1]
        dih[count+1][1] = dih[count][2]
        dih[count+1][2] = dih[count][3]

        if topology.select(f"chainid 0 and residue {i+1} and name N"):
            dih[count+1][3] = topology.select(f"chainid 0 and residue {i+1} and name N")
        #print(count, count+1)
        feature_names.append(f'phi_res{i}')
        feature_names.append(f'psi_res{i}')
        count += 2

    ## feature group 1: P-loop backbone dihedrals (Phi, Psi)
    for i in key_res['group1']: #all 6 residues in P-loop explicitly listed
        if topology.select(f"chainid 0 and residue {i - 1} and name C"):
            dih[count][0] = topology.select(f"chainid 0 and residue {i - 1} and name C")

        if topology.select(f"chainid 0 and residue {i} and name N"):
            dih[count][1] = topology.select(f"chainid 0 and residue {i} and name N")

        if topology.select(f"chainid 0 and residue {i} and name CA"):
            dih[count][2] = topology.select(f"chainid 0 and residue {i} and name CA")

        if topology.select(f"chainid 0 and residue {i} and name C"):
            dih[count][3] = topology.select(f"chainid 0 and residue {i} and name C")

        dih[count+1][0] = dih[count][1]
        dih[count+1][1] = dih[count][2]
        dih[count+1][2] = dih[count][3]

        if topology.select(f"chainid 0 and residue {i+1} and name N"):
            dih[count+1][3] = topology.select(f"chainid 0 and residue {i+1} and name N")
        #print(count, count+1)
        feature_names.append(f'phi_res{i}')
        feature_names.append(f'psi_res{i}')
        count += 2

    ## feature group 2: angle between aC and aE helices
    for i in range(3): #all 3 residues among the 85 involved in defining the angle
        if topology.select(f"chainid 0 and residue {key_res['group2'][i]} and name CA"):
            dih[count][i] = topology.select(f"chainid 0 and residue {key_res['group2'][i]} and name CA")
    if topology.select(f"chainid 0 and residue {key_res['group2'][2] - 18} and name CA"):
        dih[count][3] = topology.select(f"chainid 0 and residue {key_res['group2'][2] - 18} and name CA")
    #print(count)
    feature_names.append('aC_aE_dih')
    count += 1

    ## feature group 3: Dunbrack dihedrals
    # Phi, Psi for X-DFG, DFG-Asp, DFG-Phe
    for i in key_res['group3'][0:3]:
        if topology.select(f"chainid 0 and residue {i-1} and name C"):
            dih[count][0] = topology.select(f"chainid 0 and residue {i-1} and name C")

        if topology.select(f"chainid 0 and residue {i} and name N"):
            dih[count][1] = topology.select(f"chainid 0 and residue {i} and name N")

        if topology.select(f"chainid 0 and residue {i} and name CA"):
            dih[count][2] = topology.select(f"chainid 0 and residue {i} and name CA")

        if topology.select(f"chainid 0 and residue {i} and name C"):
            dih[count][3] = topology.select(f"chainid 0 and residue {i} and name C")

        dih[count+1][0] = dih[count][1]
        dih[count+1][1] = dih[count][2]
        dih[count+1][2] = dih[count][3]

        if topology.select(f"chainid 0 and residue {i+1} and name N"):
            dih[count+1][3] = topology.select(f"chainid 0 and residue {i+1} and name N")
        #print(count, count+1)
        feature_names.append(f'phi_res{i}')
        feature_names.append(f'psi_res{i}')
        count += 2

    # Chi1 for DFG-Phe
    #print(count)
    if topology.select(f"chainid 0 and residue {key_res['group3'][2]} and name N"):
        dih[count][0] = topology.select(f"chainid 0 and residue {key_res['group3'][2]} and name N")

    if topology.select(f"chainid 0 and residue {key_res['group3'][2]} and name CA"):
        dih[count][1] = topology.select(f"chainid 0 and residue {key_res['group3'][2]} and name CA")

    if topology.select(f"chainid 0 and residue {key_res['group3'][2]} and name CB"):
        dih[count][2] = topology.select(f"chainid 0 and residue {key_res['group3'][2]} and name CB")

    if topology.select(f"chainid 0 and residue {key_res['group3'][2]} and name CG"):
        dih[count][3] = topology.select(f"chainid 0 and residue {key_res['group3'][2]} and name CG")
    feature_names.append(f"chi1_res{key_res['group3'][2]}")


    ### distances
    ## feature group 2:
    if topology.select(f"chainid 0 and residue {key_res['group2'][3]} and name NZ"):
        dis[0][0] = topology.select(f"chainid 0 and residue {key_res['group2'][3]} and name NZ")
    if topology.select(f"chainid 0 and residue {key_res['group2'][4]} and name OE1"):
        dis[0][1] = topology.select(f"chainid 0 and residue {key_res['group2'][4]} and name OE1")

    dis[1][0] = dis[0][0]
    if topology.select(f"chainid 0 and residue {key_res['group2'][4]} and name OE2"):
        dis[1][1] = topology.select(f"chainid 0 and residue {key_res['group2'][4]} and name OE2")
    feature_names.append('KE_OE1_dis')
    feature_names.append('KE_OE2_dis')

    ## feature group 3: Dunbrack distances D1, D2
    if topology.select(f"chainid 0 and residue {key_res['group3'][3]} and name CA"):
        dis[2][0] = topology.select(f"chainid 0 and residue {key_res['group3'][3]} and name CA")
    if topology.select(f"chainid 0 and residue {key_res['group3'][2]} and name CZ"):
        dis[2][1] = topology.select(f"chainid 0 and residue {key_res['group3'][2]} and name CZ")
    if topology.select(f"chainid 0 and residue {key_res['group2'][3]} and name CA"):
        dis[3][0] = topology.select(f"chainid 0 and residue {key_res['group2'][3]} and name CA")
    dis[3][1] = dis[2][1]
    feature_names.append('Dunbrack_D1')
    feature_names.append('Dunbrack_D2')

    # DFG conformation-related distances D3, D4
    dis[4][0] = dis[2][0]
    if topology.select(f"chainid 0 and residue {key_res['group3'][1]} and name CG"):
        dis[4][1] = topology.select(f"chainid 0 and residue {key_res['group3'][1]} and name CG")
    dis[5][0] = dis[3][0]
    dis[5][1] = dis[4][1]
    feature_names.append('DFG_D3')
    feature_names.append('DFG_D4')

    ## feature group 4: FRET distance
    if topology.select(f"chainid 0 and residue {key_res['group4'][0]} and name CA"):
        dis[6][0] = topology.select(f"chainid 0 and residue {key_res['group4'][0]} and name CA")
    if topology.select(f"chainid 0 and residue {key_res['group4'][1]} and name CA"):
        dis[6][1] = topology.select(f"chainid 0 and residue {key_res['group4'][1]} and name CA")
    feature_names.append(f'FRET_dis')

    '''
    ## feature group 5: A-loop "lock" compute_distances
    dist = 8
    for i in range(9, 12):
        #if topology.select(f"chainid 0 and residue {key_res['group0'][0] + i} and name CA"):
        print(i - 2)
        print(topology.select(f"chainid 0 and residue {key_res['group0'][0] + i} and name CA"))
        dis[i - 2][0] = topology.select(f"chainid 0 and residue {key_res['group0'][0] + i} and name CA")
        dis[i - 2][1] = topology.select(f"chainid 0 and residue {key_res['group0'][0] + i + dist} and name CA")
        dist -= 2

        feature_names.append(f'lock_dis{i - 9}')
    '''
    # check if there is any missing coordinates; if so, skip dihedral/distance calculation for those residues
    check_flag = 1
    for i in range(len(dih)):
        if 0 in dih[i]:
            dih[i] = [0,0,0,0]
            check_flag = 0

    for i in range(len(dis)):
        if 0 in dis[i]:
            dis[i] = [0,0]
            check_flag = 0
    if check_flag:
        print("There is no missing coordinates.  All dihedrals and distances will be computed.")

    # calculate the dihedrals and distances for the user-specifed structure (a static structure or an MD trajectory)
    dihedrals = md.compute_dihedrals(traj, dih[-7:][:])/np.pi*180
    distances = md.compute_distances(traj, dis[2:4])
    # clean up
    del traj, dih, dis
    return dihedrals, distances
