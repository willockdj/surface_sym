# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:55:21 2020

@author: dave
"""
import numpy as np
import sys
import csv
import os
#
# Function to obtain HOME
from os.path import expanduser
#
from ase import Atom
from ase import Atoms

from ase.optimize import BFGS
from ase.optimize import FIRE
from ase.constraints import FixAtoms
from ase.constraints import FixBondLength

from ase.io import read, write, Trajectory

from ase.calculators.vasp import Vasp2

home = expanduser("~")
set_module="%s/python/modules/slab" % home
sys.path.append(set_module)
set_module="%s/python/modules/atom_settings" % home
sys.path.append(set_module)
set_module="%s/python/modules/num_digi" % home
sys.path.append(set_module)
set_module="%s/python/modules/vectors" % home
sys.path.append(set_module)
#
print("Current system path:")
print(sys.path)
#
# Import our own modules
#
from set_atoms import atom_formal_charges
#
from binary_tools import *
from vectors import *
#
# Local definition of atom...atom distance measurement
#
def atom_dist(atoms, i1, i2):
#
#  Inter-atomic vector
#
    vec= np.array([ atoms.positions[i2,0]-atoms.positions[i1,0],    \
                    atoms.positions[i2,1]-atoms.positions[i1,1],    \
                    atoms.positions[i2,2]-atoms.positions[i1,2] ])
#
#  impose minimum image convention
#
    latt      = (atoms.get_cell()).copy()
    recip_latt= (atoms.get_reciprocal_cell()).copy()
#
    print(latt)
    print(recip_latt)
#
    vec= np.array([ atoms.positions[i2,0]-atoms.positions[i1,0],    \
                    atoms.positions[i2,1]-atoms.positions[i1,1],    \
                    atoms.positions[i2,2]-atoms.positions[i1,2] ])

    size = np.sqrt(np.vdot(vec,vec))
    vec = vec/size
    
    return size
#
# Function to find and record layers within a structure assuming
# z-axis is perpendicular to surface
#
def find_layers(atoms, tol):

   z_this=atoms.positions[0][2]
   layers=[]
   flag=np.ones(len(atoms))
   first=True
#
   while np.sum(flag) != 0:
      this_layer=[]
      if first:
        first=False
        this_layer.append(0)
        flag[0]=0
#
      for iatom in range(1,len(atoms)):
         if (flag[iatom]==1):
            if (   (atoms.positions[iatom][2] > (z_this - tol)) \
                 & (atoms.positions[iatom][2] < (z_this + tol)) ):
               flag[iatom]=0
               this_layer.append(iatom)
        
      layers.append(this_layer)
# find next z
      for iatom in range(1,len(atoms)):
         if (flag[iatom]==1):
            z_this=atoms.positions[iatom][2]
            break
#
   return layers      
    
#
# Main code begins
#
# debug: setting "True" allows checking of code without calling fhiaims
#        this can be run in the foreground to check set up before 
#        commiting to a fhiaims calculation. Useful here for magnetic
#        ordering check.
#        setting "False" will run the fhiaims jobs so only do as part of a
#        job in a queue
#
debug = True 
#
if (debug):
  print(f"DEBUG version, checking code, fhiaims will not be run")
  print(f"DEBUG foreground execution with additional printing")
#
  workdir = os.getcwd()                                           # Directory where we will do the calculations
  geom_file=sys.argv[1]                                           # structure file will be passed to script
  cores_per_task = "not defined in debug mode"                    # cores assigned to tasks where available
  jobid = "not defined in debug mode"                             # Unique identifier for this job
  stdout_file= "not defined in debug mode"                        # stdout file will appear in your launch directory
  results_dir= os.getcwd()                                        # directory for results, this should be on your $HOME space
  traj_file = "not defined in debug mode"                         # File for trajectory output
  traj_spin_file = "not defined in debug mode"                    # File for trajectory output
  machine = "not defined in debug mode"                           # string defining machine
else:
  print(f"Arguements passed {len(sys.argv)}")
  for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>7}: {arg}")
#
# this ase_vasp_opt.py script expects:
# the first arguement to be the directory where we will run vasp, 
# second the number of cores available or a junk string if this is not required
# The third to be the JOB_ID, a unique identifier to add to output information
# The fourth is the path to the directory from which the job was launched
# The fifth is the sub-directory for the structure for this particular run.
# The sixth is the sub-directory for the sub_directory for this type of job, under the structure
# The seventh to be the name of the machine we are running on.
#           Currently this can be "hawk","thomas" or  "archer".
#
#
  workdir = sys.argv[1]                                           # Directory where we will do the calculations
  cores_per_task = sys.argv[2]                                    # cores assigned to tasks where available
  jobid = sys.argv[3]                                             # Unique identifier for this job
  stdout_file= "%s/vasp_%s_%s.stdout" % (sys.argv[4], sys.argv[5], jobid) # stdout file for fhiaims will appear in your launch directory
  results_dir= "%s/%s/%s" % (sys.argv[4], sys.argv[5], sys.argv[6]) # directory for results, this should be on your $HOME space
  traj_file = "%s/%s_%s_spin1.traj" % ( results_dir, sys.argv[5], jobid )      # File for trajectory output
  traj_spin_file = "%s/%s_%s_spin2.traj" % ( results_dir, sys.argv[5], jobid ) # File for trajectory output
  machine = sys.argv[7].lower()                                     # string defining machine
#
print(f"------------------------------------------------------------")
print(f"Work directory               : %s" % workdir )
print(f"Geometry file                : %s" % geom_file )
print(f"stdout file for fhiaims      : %s" % stdout_file )
print(f"results_dir for this script  : %s" % results_dir )
print(f"trajectory file for this run : %s" % traj_file )
print(f"trajectory file for ISPIN 2  : %s" % traj_spin_file )
print(f"Machine running job          : %s" % machine )
print(f"------------------------------------------------------------")
#
# set command for running fhiaims, stdout file destination
# This is machine dependent, for a new machine add what would have been in
# the job control script to launch the calculation.
#
if 'hawk' in machine:
     cmd_string = "mpirun -n %s vasp_std > %s" % (cores_per_task, stdout_file)
elif 'thomas' in machine:
     cmd_string = "gerun vasp_gam > %s" % stdout_file
elif 'archer' in machine:
     cmd_string = "aprun -n $NPROC vasp_gam > %s" % stdout_file
else:
     if (debug):
        print(f"DEBUG: No command for fhiaims required or set.")
     else:
        print(f"ERROR: Unknown machine, do not know command for running fhiaims!")
#
# Read in the structure from a cif file   
#
atoms=read(geom_file)
write('check_start.cif', atoms ,format='cif')
write('check_start.xyz', atoms ,format='xyz')

#
# Set atom charges
# Formal charges just used to estimate the slab dipole
#
totq = atom_formal_charges(atoms)
#
zdip = 0
for iatom in range(0,len(atoms)):
      zdip += atoms[iatom].charge * atoms.positions[iatom][2] 
#
# Try find_layers
#
layers=find_layers(atoms,0.1)
for ilayer in range(0,len(layers)):
   print("layer %d" % ilayer )
   for iatom in range(0,len(layers[ilayer])):
      indx=layers[ilayer][iatom]
      print("%d %s %10.6f %10.6f %10.6f " % ( indx, atoms.symbols[indx], \
             atoms.positions[indx][0], atoms.positions[indx][1],         \
             atoms.positions[indx][2] ))
#
# In this example the top layer (7) is already H in hcp sites
# Use layer (4) to place this atoms exactly in hollow center
#
top_hcp=[layers[7][i] for i in range(0,len(layers[7]))]
#
for iatom in range(0,len(top_hcp)):
   atoms.positions[top_hcp[iatom]][0]=atoms.positions[layers[4][iatom]][0]
   atoms.positions[top_hcp[iatom]][1]=atoms.positions[layers[4][iatom]][1]
#
# Use layer (5) to define the fcc sites
# First copy top layer for H atributes
#
top_hcp=[layers[7][i] for i in range(0,len(layers[7]))]
#
indx=len(atoms)
top_fcc=[indx+i for i in range(0,len(layers[5]))]
#
# add the fcc hollow atoms by copying these to get H attributes 
# but take co-ordinats from layer 5(x,y).
#
for iatom in range(0,len(layers[7])):
   atoms.append(atoms[layers[7][iatom]])
#
for iatom in range(0,len(layers[5])):
   atoms.positions[indx][0]=atoms.positions[layers[5][iatom]][0]
   atoms.positions[indx][1]=atoms.positions[layers[5][iatom]][1]
   indx+=1
#
print("added")
for iatom in range(0,len(top_fcc)):
    indx=top_fcc[iatom]
    print("%d %s %10.6f %10.6f %10.6f " % ( indx, atoms.symbols[indx], \
             atoms.positions[indx][0], atoms.positions[indx][1],         \
             atoms.positions[indx][2] ))
#
write('full_cover.cif', atoms, format='cif')
write('full_cover.in', atoms, format='aims')
#
# Top layer now has two distinct sites defined by top_hcp and top_fcc index arrays
#
num_top_hcp=len(top_hcp)
num_top_fcc=len(top_fcc)
num_top_tot=num_top_hcp+num_top_fcc
#
sys.exit("Just setting up")
#
# work out possible patterns, in current test have 18 sites in all so 9 removals will
# give large number of configurations
#
for num_to_set in range(2,3):
#
# Need to deal with the hcp and fcc sites as distinct. So
# loop through the possible division of adsorbates between the sites.
# loop over possible hcp populations and then set fcc as the differnce
# of these and the total required.
# 
   nconfigs=0
   nunique_tot=0
#
   for num_to_set_hcp in range(0,num_to_set+1):
#
# rest must be fcc site atoms
#
      num_to_set_fcc=num_to_set-num_to_set_hcp
#
      print(" ")
      print(" ")
      print("Will set %d of 18 with %d hcp and %d fcc" % (num_to_set, num_to_set_hcp, num_to_set_fcc))
#
# The arrangements can be taken separately for hcp and fcc sites. But for each option 
# on the hcp lattice will have to test all fcc possibilities. Hence total configurations
# is the product of the two.
#
      flag_list_hcp= k_bits_on(num_to_set_hcp,num_top_hcp)
      flag_list_fcc= k_bits_on(num_to_set_fcc,num_top_fcc)
      nlist_hcp=len(flag_list_hcp)
      nlist_fcc=len(flag_list_fcc)
      nlist_tot=nlist_hcp*nlist_fcc
      nconfigs+=nlist_tot
#
      print("This will require %d hcp and %d fcc to be considered, altogether %d configurations" %
                                                            (nlist_hcp, nlist_fcc, nlist_tot))
#
# Now need to split the dists list into those within a set and those between sets
#
      all_dists_hcp=[]
      all_dists_fcc=[]
      all_dists_htf=[]
#
      print
      print("Setting up ", num_to_set, "vacancies, need to test ", nlist_tot, " arrangements.")
      print
#
# Work out number to expect in dist lists, this is the number of pair distances for that
# arrangement. Again the dist lists need the distances within each set and the distances between
# sets. Within sets we need to avoid double counting, but between sets we do not.
#
# The dist_lists contain the indicies for each atom pair, so for the intra-set lists
# the first entry is for atom 0 and so is a list of all others, the second is for 
# atom 1 and ignores 0 and 1 etc.
# For the inter-set lists we need all the indicies of the other list in each case.
#
      num_in_hcp_dist_list=0
      num_in_fcc_dist_list=0
      num_in_htf_dist_list=0
#
      dist_list_hcp=[]
      dist_list_fcc=[]
      dist_list_htf=[]
#
      unique_hcp_list=[]
      unique_fcc_list=[]
      unique_htf_list=[]
      num_of_unique=[]
#
      for ilist in range(0,num_to_set_hcp):
         num_in_hcp_dist_list += ilist
         if (ilist < num_to_set_hcp-1):
           dist_list_hcp.append([j for j in range(ilist+1,num_to_set_hcp)])
#
# Note that in later code all the H sites that are to be deleted will be
# placed in a single list and this will be done hcp followed by fcc so that
# the fcc site indices follow the hcp.
#
      for ilist in range(0,num_to_set_fcc):
         num_in_fcc_dist_list += ilist
         if (ilist < num_to_set_fcc-1):
           dist_list_fcc.append([j+num_to_set_hcp for j in range(ilist+1,num_to_set_fcc)])
#
# No double counting in the hcp to fcc list
      num_in_htf_dist_list=num_to_set_hcp*num_to_set_fcc
#
# Now we need a list of all the fcc sites for each hcp site, so that all inter-vacancy 
# distances will be obtained.
#
      for ilist in range(0,num_to_set_hcp):
           dist_list_htf.append([j+num_to_set_hcp for j in range(0,num_to_set_fcc)])
#
#
      print("num_in_hcp_dist_list = ",num_in_hcp_dist_list)
      for ilist in range(0,num_to_set_hcp-1):
         print(dist_list_hcp[ilist])
#
      print("num_in_fcc_dist_list = ",num_in_fcc_dist_list)
      for ilist in range(0,num_to_set_fcc-1):
         print(dist_list_fcc[ilist])

      print("num_in_htf_dist_list = ",num_in_htf_dist_list)
      for ilist in range(0,num_to_set_hcp):
         print(dist_list_htf[ilist])

      hcp_dists=np.empty([num_in_hcp_dist_list])
      fcc_dists=np.empty([num_in_fcc_dist_list])
      htf_dists=np.empty([num_in_htf_dist_list])
#
      num_dists=0 
#
# -------------------------------------------------------------
# Create flag lists for the configurations on each sub-set
# Main look for structure creation, now a nested double loop.
# -------------------------------------------------------------
#
      print("Starting loops for this hcp/fcc population setting, first pass set")
      first_pass=True                                      # note first pass of fcc loop
#
      for ihcp in range(0,nlist_hcp):
         hcp_struct_id =  flags_to_int(flag_list_hcp[ihcp])
         print(flag_list_hcp[ihcp],"hcp flag list : ", hcp_struct_id)
#
# Nested so that we consider every configuration on fcc with each hcp arrangement
#
         for ifcc in range(0,nlist_fcc):
            fcc_struct_id =  flags_to_int(flag_list_fcc[ifcc])
            print(flag_list_fcc[ifcc],"fcc flag list : ", fcc_struct_id)
#
            this_atoms=atoms.copy()
            this_vacs=atoms.copy()
#
# Remove num_to_set H atoms from top of slab
#
            flag=[]
            for iatom in range(0,num_top_hcp):  
               if ( flag_list_hcp[ihcp][iatom] == 1 ): 
                   flag.append(top_hcp[iatom])
#
            for iatom in range(0,num_top_fcc):  
               if ( flag_list_fcc[ifcc][iatom] == 1 ): 
                   flag.append(top_fcc[iatom])
#  
# delete the atoms from the hydrogen layer
#
            del this_atoms[flag]
#
# Make top_vacs as atom set for just the atoms about to be deleted
#
            flag=[]
            for iatom in range(0,len(atoms)):  
               keep=-1
               for jatom in range(0,num_top_hcp):  
                  if ( iatom == top_hcp[jatom] and flag_list_hcp[ihcp][jatom] == 1 ): 
                      keep=iatom
#
               for jatom in range(0,num_top_fcc):  
                  if ( iatom == top_fcc[jatom] and flag_list_fcc[ifcc][jatom] == 1 ): 
                      keep=iatom
#
               if (keep < 0):
                  flag.append(iatom)
#
# delete all except the H atoms that will be removed in the real structure
# this allows us to look at the patterns more easily but will not be required
# for the calculations. This atom list will be used for calculating the dist lists
# that are needed to remove repeat configurations later.
#
            del this_vacs[flag]
#
# Work out the distance lists for this configuration
#
            dists_hcp=[]
            for ilist in range(0,num_to_set_hcp-1):
                dists_hcp=np.append(dists_hcp,this_vacs.get_distances(ilist,dist_list_hcp[ilist],mic=True))
#
            dists_fcc=[]
            fcc_start=num_to_set_hcp
            for ilist in range(0,num_to_set_fcc-1):
                dists_fcc=np.append(dists_fcc,this_vacs.get_distances(fcc_start+ilist,dist_list_fcc[ilist],mic=True))
#
            dists_htf=[]
            if (num_to_set_fcc > 0):
               for ilist in range(0,num_to_set_hcp):
                   dists_htf=np.append(dists_htf,this_vacs.get_distances(ilist,dist_list_htf[ilist],mic=True))
#
            print(" ")
            print("dists_hcp: ", dists_hcp)
            print("dists_fcc: ", dists_fcc)
            print("dists_htf: ", dists_htf)
#
# Compare this set of distances with those already seen
# num_dists will hold the number of unique distance patterns found so far.
#
            if (first_pass):
               first_pass=False
               all_dists_hcp = dists_hcp
               all_dists_fcc = dists_fcc
               all_dists_htf = dists_htf
               num_dists=1
               unique_list=[[hcp_struct_id, fcc_struct_id]]
               num_of_unique=[1]

               print("First pass all_dists_hcp : ",all_dists_hcp)
               print("First pass all_dists_fcc : ",all_dists_fcc)
               print("First pass all_dists_htf : ",all_dists_htf)

               vac_file="top_vac_atoms_%d_%d_%d.cif" % (num_to_set, hcp_struct_id, fcc_struct_id)
               write(vac_file, this_vacs ,format='cif')
#
# Write the slab with defects present
               def_slab_file="def_slab_%d_%d_%d.cif" % (num_to_set, hcp_struct_id, fcc_struct_id)
               write(def_slab_file, this_atoms,format='cif')
               write(def_slab_file, this_atoms,format='aims')
#
# Need to test to see if this configuration has been seen before by comparing the
# dist arrays with those already seen
#
            else:
               ind_hcp=0
               ind_fcc=0
               ind_htf=0
               matched=False
#  
#  Loop over the dist_lists that we already have
#  
               for which_dist in range(0,num_dists):
                  test_dist_hcp=[]
                  test_dist_fcc=[]
                  test_dist_htf=[]
#  
#  copy the next distribution from the full list into the test_dist arrays
#  
#                  print("Checking against hcp_dist list with %d members" % num_in_hcp_dist_list)
                  for jdist in range(0,num_in_hcp_dist_list):
#                     print("ind_hcp = %d" % ind_hcp)
                     test_dist_hcp.append(all_dists_hcp[ind_hcp])
                     ind_hcp += 1
#  
#                  print("Checking against fcc_dist list with %d members" % num_in_fcc_dist_list)
                  for jdist in range(0,num_in_fcc_dist_list):
#                     print("ind_fcc = %d" % ind_fcc)
                     test_dist_fcc.append(all_dists_fcc[ind_fcc])
                     ind_fcc += 1
#  
#                  print("Checking against htf_dist list with %d members" % num_in_htf_dist_list)
                  if (num_to_set_fcc > 0):
                     for jdist in range(0,num_in_htf_dist_list):
#                        print("ind_htf = %d" % ind_htf)
                        test_dist_htf.append(all_dists_htf[ind_htf])
                        ind_htf += 1
#
#                  sys,exit("Just testing")
#
# To have seen before all three lists need to match 
#
                  matched_hcp = num_in_hcp_dist_list == 0 or match_vectors(dists_hcp,test_dist_hcp)
                  matched_fcc = num_in_fcc_dist_list == 0 or match_vectors(dists_fcc,test_dist_fcc)
                  matched_htf = num_in_htf_dist_list == 0 or match_vectors(dists_htf,test_dist_htf)
#
                  if ( matched_hcp and  matched_fcc and  matched_htf ):
                     matched=True
                     print("Matched with existing list...")
                     num_of_unique[which_dist] += 1
                     break
#
# If this has not been matched must have a new unique configuration
#
               if (not matched):
                  num_dists+=1
                  all_dists_hcp=np.append(all_dists_hcp,[dists_hcp])
                  all_dists_fcc=np.append(all_dists_fcc,[dists_fcc])
                  all_dists_htf=np.append(all_dists_htf,[dists_htf])
                  unique_list.append([hcp_struct_id,fcc_struct_id])
                  num_of_unique.append(1)
                  print("New unique configuration adding to lists")
                  print("Now all_dists_hcp : ",all_dists_hcp)
                  print("Now all_dists_fcc : ",all_dists_fcc)
                  print("Now all_dists_htf : ",all_dists_htf)
#
                  vac_file="top_vac_atoms_%d_%d_%d.cif" % (num_to_set, hcp_struct_id, fcc_struct_id)
                  write(vac_file, this_vacs ,format='cif')
#
# Write the slab with defects present
                  def_slab_file="def_slab_%d_%d_%d.cif" % (num_to_set, hcp_struct_id, fcc_struct_id)
                  write(def_slab_file, this_atoms ,format='cif')
                  write(def_slab_file, this_atoms ,format='aims')

      nunique_tot+=num_dists
#
# End of loop
#
   print(" ")
   print(f"Thats %d patterns altogether of which %d are unique" % (nconfigs, nunique_tot))
#
# Report the data
#
   csv_headers=['index','hcp structure index', 'fcc structure index', 'Number occurances']
   csv_filename="degen_%d.csv" % num_to_set
   csv_file = open(csv_filename, 'w', newline='')
   csv_writer = csv.writer(csv_file)
   csv_writer.writerow(csv_headers)

   for ind in range(0, num_dists):
#
      csv_row = [ind, unique_list[ind], num_of_unique[ind]]
#
      csv_writer.writerow(csv_row)
#
   csv_file.close()
#
   print(f"End of program")
#
