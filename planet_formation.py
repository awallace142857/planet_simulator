from __future__ import division
import numpy as np,matplotlib.pyplot as plt,math,scipy.ndimage as nd
import astropy.io.fits as pyfits
import os,sys
from PIL import Image
from matplotlib import ticker
import aplpy
import matplotlib.cm as cm
import astropy.constants as c, astropy.units as u
inDir = 'fitting' #Directory Containing Mass, Radius, Entropy and Luminosity trends from MESA and other fits
def twoD_conv(f,g):
	"""Convolution between two matrices"""
	f_FFT = np.fft.rfft2(f)
	g_FFT = np.fft.rfft2(g)
	prod = f_FFT*g_FFT
	return np.fft.fftshift(np.fft.irfft2(prod))
	
def cross_corr(f,g):
	"""Cross-correlation between two matrices"""
	f_FFT = np.fft.rfft2(f)
	g_FFT = np.fft.rfft2(g)
	prod = g_FFT*np.conj(f_FFT)
	return np.fft.fftshift(np.fft.irfft2(prod))

def find_entropy(mass,radius):
	"""Calculates the central entropy of a planet using fits from MESA data
	
	Parameters
	----------
	mass: float
		Planet Mass in MJ
	radius: float
		Planet Radius in RJ
	
	Returns
	-------
	entropy: float
		The planet's central entropy in k_B/baryon
	"""
	global inDir
	infile = open(inDir+'/entropyRadii','r')
	left = 0.1
	ms = []
	slopes = []
	intercepts = []
	S_maxs = []
	for line in infile:
		entry = line.split(' ')
		ms.append(float(entry[0]))
		slopes.append(float(entry[1]))
		intercepts.append(float(entry[2]))
		S_maxs.append(float(entry[3]))
	ms = np.array(ms)
	slopes = np.array(slopes)
	intercepts = np.array(intercepts)
	S_maxs = np.array(S_maxs)
	slopeFit = np.polyfit(np.log(ms),slopes,1)
	intFit = np.polyfit(np.log(ms),intercepts,1)
	maxFit = np.polyfit(np.log(ms),S_maxs,1)
	slope = slopeFit[0]*np.log(mass)+slopeFit[1]
	intercept = intFit[0]*np.log(mass)+intFit[1]
	S_max = maxFit[0]*np.log(mass)+maxFit[1]
	entropy = S_max-np.exp(slope*np.log(np.log(radius/left))+intercept)
	return entropy
	
def find_radius(mass,entropy):
	"""Calculates a planet's radius for given mass and central entropy using MESA data
	
	Parameters
	----------
	mass: float
		Planet Mass in MJ
	entropy: float
		Planet Central Entropy in k_B/baryon
	
	Returns
	-------
	radius: float
		The planet's radius in RJ
	"""
	global inDir
	infile = open(inDir+'/fitParams','r')
	entropies = []
	coeffs = []
	powers = []
	minX = []
	minY = []
	for line in infile:
		if 'Entropy' in line:
			continue
		entry = line.split(' ')
		entropies.append(float(entry[0]))
		coeffs.append(float(entry[1]))
		powers.append(float(entry[2]))
		minX.append(float(entry[3]))
		minY.append(float(entry[4]))
	entropies = np.array(entropies)
	coeffs = np.array(coeffs)
	powers = np.array(powers)
	minX = np.array(minX)
	minY = np.array(minY)
	rads = []
	for ii in range(len(entropies)):
		rads.append(coeffs[ii]*(np.log10(mass/minX[ii]))**powers[ii]+minY[ii])

	rads = np.array(rads)
	left = 0.8*min(rads)
	top = 1.2*max(entropies)
	fit = np.polyfit(np.log(rads[np.isfinite(rads)]-left),np.log(-entropies[np.isfinite(rads)]+top),1)
	const = 0.8*min(minX)
	minXCoeffs = np.polyfit(entropies,np.log(minX-const),1)
	left = np.exp(minXCoeffs[1]+minXCoeffs[0]*entropy)+const
	const = 0.8*min(minY)
	minYCoeffs = np.polyfit(entropies,np.log(minY-const),1)
	bottom = np.exp(minYCoeffs[1]+minYCoeffs[0]*entropy)+const
	fitCoeff = np.polyfit(entropies,coeffs,1)
	fitPower = np.polyfit(entropies,powers,1)
	a = fitCoeff[1]+fitCoeff[0]*entropy
	p = fitPower[1]+fitPower[0]*entropy
	radius = a*(np.log10(mass/left))**p+bottom
	return radius
	
def luminosity_evolution(mass,radius,t_vec):
	"""Calculates the planet's luminosity over time using MESA fits
	
	Parameters
	----------
	mass: float
		Planet Mass in MJ
	radius: float
		Planet Radius in RJ
	t_vec: array
		The time vector in Myr
	
	Returns
	-------
	[lum,slope,offset,intercept]: [array,array,array,array]
		A list containing the planet's luminosity in L_sun
		as well as the fit parameters each as a function of time
	"""
	global inDir
	radStrings = []
	radFloats = []
	allFiles = os.listdir(inDir)
	for file in allFiles:
		if 'params' in file and '.' in file:
			for ii in range(len(file)):
				if file[ii]=='s':
					rad = file[ii+1:len(file)]
					if rad not in radStrings:
						infile = open(inDir+'/params'+rad,'r')
						nLines = 0
						for line in infile:
							nLines+=1
						if nLines>12:
							radStrings.append(rad)
							radFloats.append(float(rad))
					break
	diffs = np.zeros(len(radFloats))
	for jj in range(len(diffs)):
		diffs[jj] = np.abs(radFloats[jj]-radius)
	rad = radStrings[diffs.argmin()]
	infile = open(inDir+'/params'+rad,'r')
	ms = []
	slopes = []
	intercepts = []
	offsets = []
	for line in infile:
		if 'Mass' not in line:
			entry = line.split(' ')
			ms.append(float(entry[0]))
			slopes.append(float(entry[2]))
			intercepts.append(float(entry[3]))
			offsets.append(float(entry[4]))
	ms = np.array(ms)
	slope = slopes[0]
	intercepts = np.array(intercepts)
	offsets = np.array(offsets)
	order = 10
	offCoeffs = np.polyfit(ms,offsets,order)
	offset = 0
	for p in range(order+1):
		offset+=offCoeffs[order-p]*mass**p
	intCoeffs = np.polyfit(np.log(ms),intercepts,1)
	intercept = intCoeffs[0]*np.log(mass)+intCoeffs[1]
	lum = np.exp(slope*np.log(t_vec-offset)+intercept)
	return [lum,slope,offset,intercept]

def mass_sep_dist(mass_min,mass_max,sep_min,sep_max,mass_power,sep_power1,sep_power2,const1,cutoff):
	"""Randomly generates a planet's final mass and orbital radius
	using a power law distribution.  
	dN/(dlnMdlnr) = const*M**(mass_power)*r**(sep_power)
	The orbital radius power changes at 10AU but the mass power 
	is constant across the distribution.
	
	Parameters
	----------
	mass_min: float
		Lower limit on mass in MJ
	mass_max: float
		Upper limit on mass in MJ
	sep_min: float
		Lower limit on orbital radius in AU
	mass_max: float
		Upper limit on orbital radius in AU
	mass_power: float
		Power used for the mass dependence
	sep_power1: float
		Power used for the orbital radius dependence <=10AU
	sep_power2: float
		Power used for the orbital radius dependence >10AU
	const1: float
		Normalization constant below 10AU.  Above 10AU, a new
		constant is calculated such that the distribution is continuous
	
	Returns
	-------
	[mass,sep]: [float,float]
		A list with a random mass in MJ as the first element and orbital
		radius in AU as the second element
	"""
	const2 = const1*cutoff**(sep_power1-sep_power2)
	nside = 100
	boxes = []
	area_bounds = []
	count = 0
	total_int = 0
	for ii in range(nside):
		for jj in range(nside):
			bound_mass = [np.log(mass_min)+jj*(np.log(mass_max/mass_min)/nside),np.log(mass_min)+(jj+1)*(np.log(mass_max/mass_min)/nside)]
			bound_sep = [np.log(sep_min)+ii*(np.log(sep_max/sep_min)/nside),np.log(sep_min)+(ii+1)*(np.log(sep_max/sep_min)/nside)]
			if bound_sep[0]<=np.log(cutoff):
				power = sep_power1
				k = const1
			else:
				power = sep_power2
				k = const2
			if power == 0:
				integral = (k/(mass_power))*(np.exp(mass_power*bound_mass[1])-np.exp(mass_power*bound_mass[0]))*(bound_sep[1]-bound_sep[0])
			else:
				integral = (k/(mass_power*power))*(np.exp(mass_power*bound_mass[1])-np.exp(mass_power*bound_mass[0]))*(np.exp(power*bound_sep[1])-np.exp(power*bound_sep[0]))
			total_int+=integral
			boxes.append([bound_mass[0],bound_mass[1],bound_sep[0],bound_sep[1]])
			if count==0:
				area_bounds.append([0,integral])
			else:
				area_bounds.append([area_bounds[count-1][1],area_bounds[count-1][1]+integral])
			count+=1
	rand_num = np.random.random()
	for ii in range(len(area_bounds)):
		if rand_num>=area_bounds[ii][0]/total_int and rand_num<area_bounds[ii][1]/total_int:
			used = ii
			break
	mass = np.exp(boxes[used][0]+(boxes[used][1]-boxes[used][0])*np.random.random())
	sep = np.exp(boxes[used][2]+(boxes[used][3]-boxes[used][2])*np.random.random())
	return [mass,sep]
	
def zhu(radius,mmDot,band):
	"""Calculates a planet's magnitude during accretion using Zhu 2015.
	This takes data from the text file 'zhu' and calculates trends in 
	magnitude v mass*accretion rate for different planetary radii.
	
	Each line in the file corresponds to a mass*accretion rate.  For each
	line, the magnitude is assumed to vay linearly with radius.  This
	determines what the magnitude would be for the input radius.  The 
	magnitude is assumed to vary logarithmically with mass*accretion rate
	so this is fit accordingly.
	
	Parameters
	----------
	radius: float
		The planet's current radius in RJ
	mmDot: float
		The planets current mass multiplied by the current accretion rate
		mmDot is in MJ^2/yr
	band: str
		The wavelength band of the magnitude
		At the moment, only 'L' and 'K' are possible
	
	Returns
	-------
	fit: float
		The magnitude of this planets
	"""
	zhuFile = open('zhu','r')
	order = 1
	rates = []
	lums = []
	rads = np.array([1,1.5,2,4])
	if band=='L':
		offset = len(rads)
	elif band=='K':
		offset = 0
	elif band=='M':
                offset = 2*len(rads)
        elif band=='N':
                offset = 3*len(rads)
	for line in zhuFile:
		entry = line.split(' ')
		rates.append(float(entry[0]))
		if radius in rads:
			index = np.where(rads==radius)[0][0]
			lums.append(float(entry[index+offset+1]))
		else:
			radLums = []
			for ii in range(1+offset,1+offset+len(rads)):
				radLums.append(float(entry[ii]))
			radfit = np.polyfit(rads,np.array(radLums),1)
			lums.append(radfit[0]*radius+radfit[1])
	rates = np.array(rates)
	lums = np.array(lums)
	curvefit = np.polyfit(np.log10(rates),lums,order)
	fit = 0
	for ii in range(0,order+1):
		fit+=curvefit[ii]*(np.log10(mmDot))**(order-ii)
	return fit

def disk_limited(mass,c):
	"""Calculates a planet's accretion rate as a function of mass during and
	after runaway accretion.  This uses the parabola from Lissauer 2009
	and calculates accretion rate in terms of 
	disk surface density*(orbital radius)**2/period.
	
	The coefficients are calculated from fitting for high, medium and low viscosity
	respectively.
	
	Parameters
	----------
	mass: float
		The planet's current mass in MJ
	c: int
		Which set of coefficients (therefore which viscosity) to use (0,1 or 2)
	
	Returns
	-------
	10**logFit: float
		Accretion rate divided by [disk surface density*(orbital radius)**2/period]
	"""
	coeffs = [[-1.28161436,-1.62527386,-2.86926726],[-1.59541038,-2.7974245,-4.26371995],[-1.91234071,-3.54530977,-4.72537617]]
	"""logFit1 = coeffs[0]*(np.log10(mass/1000))**2+coeffs[1]*np.log10(mass/1000)+coeffs[2]
	coeffs = [-1.23,-8.97,-18.67]"""
	logFit = coeffs[c][0]*(np.log10(mass))**2+coeffs[c][1]*np.log10(mass)+coeffs[c][2]
	return 10**logFit

def rad_time(time,mass):
	"""Calculates a planet's radius as it contracts over time using trends
	from Spiegel & Burrows 2012.  This uses data from the text file 
	'find_lum' which contains radius and temperature trands for 1MJ and 
	10MJ planets.
	
	The trends are calculated and expected to follow power laws.  Log(radius)
	is fit to arcsinh(time) to prevent the radius becoming infinite at t0.
	This is calculated for the two masses and the coefficients are expected 
	to vary logarithmically with mass.
	
	The new coefficients are simply the coefficients for 1MJ plus the ratio of
	log(mass) to log(10) times the diffence between the two sets of coefficients.
	These coefficients are then used to calculate the radius at the input time.
	
	Parameters
	----------
	time: float
		Planet age in Myr
	mass: float
		Planet mass in MJ
	
	Returns
	-------
	rad: float
		The resultant planet radius in RJ
	"""
	infile = open('find_lum','r')
	R1 = []
	R10 = []
	seg = 0
	for line in infile:
		entry = line.split(' ')
		if seg==0:
			if len(entry)>1:
				R1.append([float(entry[0]),float(entry[1])])
			if '10R' in line:
				seg = 1
		elif seg==1:
			if len(entry)>1:
				R10.append([float(entry[0]),float(entry[1])])
			if '1T' in line:
				seg = 2

	R1 = np.array(R1)
	R10 = np.array(R10)
	coeffsR1 = np.polyfit(np.arcsinh(R1[:,0]),np.log(R1[:,1]),1)
	coeffsR10 = np.polyfit(np.arcsinh(R10[:,0]),np.log(R10[:,1]),1)
	coeffsR = coeffsR1+(np.log(mass)/np.log(10))*(coeffsR10-coeffsR1)
	rad = np.exp(coeffsR[0]*np.arcsinh(time)+coeffsR[1])
	return rad
def extreme_mass(sep,start_time,init_mass,disk_life,period,time_power,threshold,surface_density,c):
	"""Calculates the maximum possible planet mass for given starting 
	conditions.  Using a planet's start time and initial mass, the timespan
	of the initial phase is solved for analytically.  The planet then goes 
	through rapid accretion using the disk_limited function until the disk
	dissipates.  The planet's resultant mass is the maximum possible.
	
	Parameters
	----------
	sep: float
		The Planet's orbital radius in AU
	start_time: float
		The time when the planet starts accreting gas in yr
	init_mass: float
		The mass the planet has when gas accretion begins in MJ
	disk_life: float
		The time the disk takes to dissipate in Myr
	period: float
		The planet's period in yr
	time_power: float
		The order of magnitude of the time constant given by 10**time_power years
		This determines the timescale of the first phase
	threshold: float
		The critical mass required for runaway accretion in MJ
		(Usually init_mass*7/3)
	surface_density: float
		The local disk surface density in MJ/(AU^2)
	c: int
		The coefficients to use in the disk_limited function (0,1 or 2)
	
	Returns
	-------
	mass: float
		The maximum possible mass
	"""
	t_1 = start_time+(((0.003**3)*10**time_power)/3)*(init_mass**-3-threshold**-3)
	step = 1000
	mass = threshold
	while t_1<disk_life*1e6:
		acc = surface_density*(1-t_1*1e-6/disk_life)*(sep**2)*disk_limited(mass,c)/period
		mass+=step*acc
		t_1+=step
	return mass

def burrows(mass,radius,age,band):
	"""Calculates a planet's magnitude at a given time using
	hot start models from Spiegel & Burrows 2012.  This takes 
	data stored in text file 'burrows' which has magnitudes 
	for several masses and ages and fits a trend to the
	magnitude v time plots.
	
	It starts by finding coefficients for the 1MJ and 10MJ trends.
	The slope is assumed to be independent of mass so is taken to be
	the average of these two.  The y-intercept is assumed to increase 
	logarithmically with mass and is calculated by taking the difference
	between y-intercepts and multiplying by the log of the input mass.
	
	The radius according to Spiegel & Burrows 2012 is found using the file
	'burrows_radii' and the magnitude is offset by an amount related to
	the planet's actual radius.
	
	Parameters
	----------
	mass: float
		The planet mass in MJ
	radius: float
		The planet radius in RJ
	age: float
		The planet age in Myr
	band: str
		The wavelength band of the magnitude
		At the moment, only 'L' and 'K' are possible
	
	Returns
	-------
	fit: float
		The magnitude at this particular age
	"""
	global inDir
	radText = open(inDir+'/burrows_radii','r')
	offset = luminosity_evolution(mass,radius,age)[2]
	ts = [[],[],[]]
	rs = [[],[],[]]
	el = -1
	for line in radText:
		entry = line.split(', ')
		if 'MJ' not in line:
			ts[el].append(float(entry[0]))
			rs[el].append(float(entry[1]))
		elif 'MJ' in line:
			el+=1
	slopes = []
	intercepts = []
	minR = 1.06
	for ii in range(len(ts)):
		ts[ii] = np.array(ts[ii])
		rs[ii] = np.array(rs[ii])
		coeffs = np.polyfit(np.log(ts[ii]),np.log(rs[ii]-minR),1)
		slopes.append(coeffs[0])
		intercepts.append(coeffs[1])
	ms = np.array([1,5,10])
	slopes = np.array(slopes)
	intercepts = np.array(intercepts)
	radSlopeCoeffs = np.polyfit(ms,slopes,1)
	radIntCoeffs = np.polyfit(np.log(ms),np.log(intercepts+0.3),1)
	radSlope = radSlopeCoeffs[0]*mass+radSlopeCoeffs[1]
	radIntercept = np.exp(radIntCoeffs[0]*np.log(mass)+radIntCoeffs[1]-0.3)
	radBurrows = np.exp(radSlope*np.log(age-offset)+radIntercept)+minR
	text = open(inDir+'/burrows','r')
	ages = []
	masses = []
	hotMags = []
	for line in text:
		entry = line.split(' ')
		new_entry = []
		for ii in range(len(entry)):
			if len(entry[ii])>0:
				new_entry.append(entry[ii])
		entry = new_entry
		ages.append(float(entry[1]))
		masses.append(float(entry[2]))
		if band=='L':
			hotMags.append(float(entry[7]))
		elif band=='K':
			hotMags.append(float(entry[6]))
		elif band=='M':
                        hotMags.append(float(entry[8]))
                elif band=='N':
                        hotMags.append(float(entry[9]))
	new_ages = []
	new_masses = []
	new_hot = []
	for ii in range(len(ages)):
		if ages[ii] not in new_ages:
			new_ages.append(ages[ii])
		if masses[ii] not in new_masses:
			new_masses.append(masses[ii])
			new_hot.append([])

	new_ages = np.array(new_ages)
	new_masses = np.array(new_masses)
	for ii in range(len(ages)):
		new_hot[np.where(new_masses==masses[ii])[0][0]].append(hotMags[ii])
	new_hot = np.array(new_hot)
	mags = new_hot
	refCoeffs = []
	bins = []
	for ii in range(len(mags)):
		refCoeffs.append(np.polyfit(np.log(new_ages-offset),mags[ii],1))
	for ii in range(len(mags)+1):
		if ii==0:
			if mass<=new_masses[0]:
				bin = 0
				slope = refCoeffs[0][0]
		elif ii==len(mags):
			if mass>=new_masses[len(mags)-1]:
				bin = len(mags)
				slope = refCoeffs[len(mags)-1][0]
		else:
			if mass>=new_masses[ii-1] and mass<=new_masses[ii]:
				bin = ii
				slope = np.mean([refCoeffs[ii][0],refCoeffs[ii-1][0]])
				
	ratio = new_masses[len(mags)-1]/new_masses[0]
	diff = refCoeffs[len(mags)-1][1]-refCoeffs[0][1]
	newDiff = diff*np.log10(mass/new_masses[0])/np.log10(ratio)
	coeffs = np.zeros(2)
	coeffs = np.array([slope,refCoeffs[0][1]+newDiff])
	fit = coeffs[0]*np.log(age-offset)+coeffs[1]-5*np.log10(radius/radBurrows)
	return(fit)

def local_density(m_disk,r_disk,separation,density_power):
	"""Calculates the gas surface density at a given distance from star.
	Assumes the disk is circular, has finite radius and surface density
	proportional to r^p where p<0 and >-2.
	
	Parameters
	----------
	m_disk: float
		The disk's total gas mass in MJ
	r_disk: float
		Thsi disk's radius in AU
	separation: float
		The distance from the star in AU
	delsity_power: float
		The exponent in the power law (must be greater than -2 and less than 0)
	
	Returns
	-------
	density: float
		The gas surface density in MJ/AU^2
	"""
	if density_power<=-2 or density_power>0:
		print('ERROR! density_power must be between -2 and 0')
		sys.exit()
	density = ((density_power+2)*m_disk/(2*np.pi*r_disk**(density_power+2)))*separation**density_power
	return density
sz = 192 #The resultant image will be sz x sz pixels
dist = 140 #The distance from Earth in pc
m_star = 1 #Mass of star in solar masses
G = 0.04 #gravitational constant in (AU^3)(MJ^-1)(yr^-2)
disk_mass = 20*m_star #Disk mass in MJ
disk_extent = np.sqrt(disk_mass)*10**(1.15) #Disk radius in AU
final_age = 12 #When to stop the simulation in Myr from earliest gas accretion
disk_life = np.exp(1.1)*m_star**(-0.69) #Lifetime of Disk in Myr
step = 1000 #Step size in years
density_power1 = -1.5 #Exponent in surface density power law
min_mass = 0.3 #Minimum planet mass to be simulated (in MJ)
max_mass = 5 #Maximum planet mass to be simulated (in MJ)
min_sep = 5 #Minimum planet separation to be simulated (in AU)
max_sep = 60 #Maximum planet separation to be simulated (in AU)
mass_power = -0.31 #Exponent in mass distribution power law (from Cumming 2008)
sep_power1 = 0.39 #Exponent in separation distribution power law before cutoff (from Cumming 2008)
sep_power2 = 0 #Exponent in separation distribution power law after cutoff
cutoff = 10 #Separation at which the separation distribution changes (in AU)
const_1 = 7.2e-3 #Distribution normalisation constant if mass is in MJ and separation is in AU before cutoff
const_2 = const_1*(10**(sep_power1-sep_power2)) ##Distribution normalisation constant after cutoff
#Calculate the average number of planets per system
if sep_power2 == 0:
	planets_per_system = (1/(mass_power))*(max_mass**(mass_power)-min_mass**(mass_power))*((const_1/(sep_power1))*(cutoff**(sep_power1)-min_sep**(sep_power1))+const_2*(np.log(max_sep/cutoff)))
else:
	planets_per_system = (1/(mass_power))*(max_mass**(mass_power)-min_mass**(mass_power))*((const_1/(sep_power1))*(cutoff**(sep_power1)-min_sep**(sep_power1))+(const_2/(sep_power2))*(max_sep**(sep_power2)-cutoff**(sep_power2)))
star_mag = 0 #Absolute magnitude of star
final_entropy = 10 #Central entropy of the planet in k_B/baryon
save_dir = 'newSystems'+str(disk_mass)+'MJ'+str(cutoff)+'AUCutoff'+str(sep_power2)+'Power' #Where to save the results
samp_ages = [final_age*1e6] #At which stages to save results
#Initialise circular aperture
aperture = np.zeros((sz,sz))
for xx in range(sz):
	for yy in range(sz):
		if np.sqrt((xx-sz//2)**2+(yy-sz//2)**2)<10:
			aperture[xx,yy] = 1
if not os.path.isdir(save_dir):
	os.system('mkdir '+save_dir)
nSystems = 50000 #Number of stars to be simulated
print(planets_per_system)
for nRun in range(nSystems):
	allSeps = []
	planets = []
	allTimes = []
	allMags = []
	allRates = []
	#Calculate how many planets in this system based on the average number per system
	nPlanets = 0
	nTests = 10000
	for ii in range(nTests):
		rand = np.random.random()
		if rand<planets_per_system/nTests:
			nPlanets+=1
	if nPlanets==0:
		continue
	planets_init = [] #Initialise all existing planets
	for planet in range(nPlanets):
		start_time = step #When the planet starts accreting gas in years after time zero
		mass_sep = mass_sep_dist(min_mass,max_mass,min_sep,max_sep,mass_power,sep_power1,sep_power2,const_1,cutoff) #Calculate random mass and separation from distribution
		m_final = mass_sep[0] #Mass in MJ
		semimajor = mass_sep[1] #Separation in AU
		angle = 360*np.random.random() #Position angle in degrees
		T_disk = 280/np.sqrt(semimajor) #Local disk temperature in Kelvin
		molecular_mass = 2.35*1.673e-27/2e27 #Mean molecular mass in MJ
		stefan_boltzmann = 3.05e-58 #Stefan-Boltzmann constant in MJ(AU^2)(yr^-2)(K^-1)
		density_0 = local_density(disk_mass,disk_extent,semimajor,density_power) #Calculate initial disk surface density
		angular_speed = np.sqrt((G*m_star*1e3)/(semimajor**3)) #Angular speed in radians/yr
		period = 2*np.pi/angular_speed #Planet period in years
		sound_speed = np.sqrt(stefan_boltzmann*T_disk/molecular_mass) #Sound speed in AU/yr
		thickness = sound_speed/angular_speed #Disk thickness in AU
		time_power = 9+np.log10(np.sqrt(6./semimajor)/9+8./9) #Time constant in initial accretion function is 10^time_power years
		init_mass = 0.03 #Mass required for gas capture in MJ
		crit_mass = init_mass+init_mass/0.75 #Mass required for runaway accretion in MJ
		r_H = semimajor*(init_mass/(3*m_star*1e3))**(1./3) #Hill radius in AU
		rad = 2e3*G*init_mass/(sound_speed**2+4*G*init_mass/r_H) #Planet radius in RJ
		acc = (init_mass**4)/((10**time_power)*0.003**3) #Accretion rate in MJ/yr
		lum = G*init_mass*acc/rad
		lum_watt = lum*2e3*(1.5e11**2)*2e27*(3e7**-3) #Planet luminosity in W
		coeff = 0 #Which coefficients to use when fitting to curve in Lissauer 2009
		total_mass = extreme_mass(semimajor,start_time,init_mass,disk_life,period,time_power,crit_mass,density_0,coeff) #Maximum possible mass
		#If maximum mass is too small, reduce time_power to make the initial accretion phase shorter
		while total_mass<m_final+0.1 and time_power>5:
			time_power-=0.001
            total_mass = extreme_mass(semimajor,start_time,init_mass,disk_life,period,time_power,crit_mass,density_0,coeff)
        #If maximum mass is too large, increase time_power to make the initial accretion phase longer
        while total_mass>m_final+0.1:
			time_power+=0.001
			total_mass = extreme_mass(semimajor,start_time,init_mass,disk_life,period,time_power,crit_mass,density_0,coeff)
		total_mass = extreme_mass(semimajor,start_time,init_mass,disk_life,period,time_power,crit_mass,density_0,coeff)
		if total_mass<m_final:
			time_power-=0.001
			total_mass = extreme_mass(semimajor,start_time,init_mass,disk_life,period,time_power,crit_mass,density_0,coeff)
	#Initialise magnitude arrays
	magL = []
        magK = []
	magN = []
	magM = []
        #Store parameters in a dictionary
		if m_final<=total_mass:
			planets_init.append({'semimajor':semimajor,'age':0,'energy_loss':0,'sound_speed':sound_speed,'final_entropy':final_entropy,'max_accretion':acc,'period':period,'pa':angle,'magnitude_L':magL,'magnitude_K':magK,'magnitude_M':magM,'magnitude_N':magN,'current_mass':init_mass,'init_mass':init_mass,'final_mass':m_final,'radius':rad,'Mdot':acc,'T_disk':T_disk,'time_power':time_power,'coeff':coeff,'start_date':str(start_time),'disk_life':disk_life,'completion_date':'N/A','complete':False})
		else:
			print(m_final,total_mass,semimajor)
	#Extract all planet start times for this system
	starts = []
	for ii in range(len(planets_init)):
		starts.append(float(planets_init[ii]['start_date']))
	starts = np.array(starts)
	#Loop over time until the final age has been achieved
	for t_step in range(1,int(final_age*1e6/step)+1):
		time = t_step*step #Time in years
		diffs = abs(time-starts)
		new_planets = np.where(diffs<step/2)[0] #If close to a start time, create a new planet
		#Initialise all new planets with dictionary from before
		for ii in range(len(new_planets)):
			planets.append(planets_init[new_planets[ii]])
			allTimes.append([time/1e6])
			allMags.append([planets_init[new_planets[ii]]['magnitude_L']])
			allRates.append([planets_init[new_planets[ii]]['Mdot']])		
		for ii in range(len(planets)):
			planets[ii]['age']+=step #Increase planet age as time progresses
			planets[ii]['pa']+=(360*step/planets[ii]['period'])
			planets[ii]['pa'] = planets[ii]['pa']%360 #Change planet position accordingly but angle is always between 0 and 360
			#Decrease the disk density linearly as it dissipates
			if time<planets[ii]['disk_life']*1e6:
				surface_density = (disk_mass/(2*np.pi*disk_extent*planets[ii]['semimajor']))*(1-time*1e-6/planets[ii]['disk_life'])
			else:
				surface_density = 0
			r_H = planets[ii]['semimajor']*(planets[ii]['current_mass']/(3*m_star*1e3))**(1./3)
			if not planets[ii]['complete']:
				if planets[ii]['current_mass']-planets[ii]['init_mass']<=planets[ii]['init_mass']/0.75:
					#If the planet hasn't reached runaway accretion, accretion rate is given by M^4/(M_earth^3*10^time_power years)
					planets[ii]['Mdot'] = (planets[ii]['current_mass']**4)/((10**planets[ii]['time_power'])*0.003**3)
					planets[ii]['radius'] = 2e3*G*planets[ii]['current_mass']/(planets[ii]['sound_speed']**2+4*G*planets[ii]['current_mass']/r_H)
					#Check if the planet will reach runaway accretion by the next step.  If so, record time and mass.
					if planets[ii]['current_mass']+planets[ii]['Mdot']*step-planets[ii]['init_mass']>planets[ii]['init_mass']/0.75:
						planets[ii]['time_crit'] = time#/1e6
						planets[ii]['mass_crit'] = planets[ii]['current_mass']+planets[ii]['Mdot']*step
					lum = G*planets[ii]['current_mass']*planets[ii]['Mdot']/planets[ii]['radius']
					lum_watt = lum*2e3*(1.5e11**2)*2e27*(3e7**-3)
					planets[ii]['energy_loss']+=step*365.25*24*3600*lum_watt
					planets[ii]['magnitude_L'].append([time,1000])
					planets[ii]['magnitude_K'].append([time,1000])
					planets[ii]['magnitude_M'].append([time,1000])
					planets[ii]['magnitude_N'].append([time,1000])
				else:
					#During runaway accretion, calculate accretion rate based on Lissauer 2009
					planets[ii]['Mdot'] = surface_density*(planets[ii]['semimajor']**2)*disk_limited(planets[ii]['current_mass'],planets[ii]['coeff'])/planets[ii]['period']
					planet_rad = find_radius(planets[ii]['current_mass'],planets[ii]['final_entropy']) #Radius in RJ given by central entropy
					#Only update radius if it decreases.  Otherwise set it to 4RJ.
					if np.isfinite(planet_rad) and planet_rad<=planets[ii]['radius']:
						planets[ii]['radius'] = planet_rad
					else:
						planets[ii]['radius'] = 4
					#Magnitudes in L and K band
					planets[ii]['magnitude_L'].append([time,zhu(planets[ii]['radius'],planets[ii]['current_mass']*planets[ii]['Mdot'],'L')])
        				planets[ii]['magnitude_K'].append([time,zhu(planets[ii]['radius'],planets[ii]['current_mass']*planets[ii]['Mdot'],'K')])
					planets[ii]['magnitude_M'].append([time,zhu(planets[ii]['radius'],planets[ii]['current_mass']*planets[ii]['Mdot'],'M')])
        				planets[ii]['magnitude_N'].append([time,zhu(planets[ii]['radius'],planets[ii]['current_mass']*planets[ii]['Mdot'],'N')])
					lum = G*planets[ii]['current_mass']*planets[ii]['Mdot']/planets[ii]['radius']
					lum_watt = lum*2e3*(1.5e11**2)*2e27*(3e7**-3)
					planets[ii]['energy_loss']+=step*365.25*24*3600*lum_watt
					#Check if planet is complete
					if planets[ii]['current_mass']>=planets[ii]['final_mass'] or planets[ii]['Mdot']<1e-8 or time>=planets[ii]['disk_life']*1e6:
						planets[ii]['complete'] = True
						planets[ii]['completion_date'] = str(time)
						planets[ii]['current_mass'] = planets[ii]['final_mass']
						planet_rad = find_radius(planets[ii]['current_mass'],planets[ii]['final_entropy'])
                    	if np.isfinite(planet_rad) and planet_rad<4:
							planets[ii]['radius'] = planet_rad#1.2*planets[ii]['final_mass']**(1./3)#(1.5*planets[ii]['final_mass']/np.sqrt(planets[ii]['semimajor']/5))**(1./3)
						else:
							planets[ii]['radius'] = 4
						planets[ii]['Mdot'] = 0
						#Calculate luminosity evolution
						planets[ii]['t_vec'] = np.linspace(0,final_age-time*1e-6,(final_age*1e6-time)//step+1)
						planets[ii]['t_el'] = 0 #Element to use when calling the luminosity vector
						planets[ii]['lum_curve'] = luminosity_evolution(planets[ii]['current_mass'],planets[ii]['radius'],planets[ii]['t_vec'])[0]
						planets[ii]['LMag_curve'] = burrows(planets[ii]['current_mass'],planets[ii]['radius'],planets[ii]['t_vec'],'L')
						planets[ii]['KMag_curve'] = burrows(planets[ii]['current_mass'],planets[ii]['radius'],planets[ii]['t_vec'],'K')
						planets[ii]['MMag_curve'] = burrows(planets[ii]['current_mass'],planets[ii]['radius'],planets[ii]['t_vec'],'M')
						planets[ii]['NMag_curve'] = burrows(planets[ii]['current_mass'],planets[ii]['radius'],planets[ii]['t_vec'],'N')
				if not planets[ii]['complete']:
					planets[ii]['current_mass']+=planets[ii]['Mdot']*step #Add mass according to accretion rate
				#Find maximum accretion rate
				if planets[ii]['Mdot']>planets[ii]['max_accretion']:
					planets[ii]['max_accretion'] = planets[ii]['Mdot']
			if planets[ii]['complete']:
				lum_watt = 3.828e26*planets[ii]['lum_curve'][planets[ii]['t_el']]
				planets[ii]['magnitude_L'].append([time,planets[ii]['LMag_curve'][planets[ii]['t_el']]])
				planets[ii]['magnitude_K'].append([time,planets[ii]['KMag_curve'][planets[ii]['t_el']]])
				planets[ii]['magnitude_M'].append([time,planets[ii]['MMag_curve'][planets[ii]['t_el']]])
				planets[ii]['magnitude_N'].append([time,planets[ii]['NMag_curve'][planets[ii]['t_el']]])
				planets[ii]['t_el']+=1 #Increment element by 1
			allTimes[ii].append(time/1e6)
			allRates[ii].append(planets[ii]['Mdot'])
			allMags[ii].append(planets[ii]['magnitude_L'])
			#print(time,planets[0]['current_mass'],planets[0]['magnitude'],planets[0]['Mdot'],planets[0]['radius'],planets[0]['semimajor'])
		if len(planets)==0:
			continue
		if not time in samp_ages:
			continue
		#Create contrast ratio map of planets
		star_sim = np.zeros((sz,sz))
		star_sim[sz//2,sz//2] = 1
		for ii in range(0,len(planets)):
			contrast = 10**(-0.4*(planets[ii]['magnitude_L'][len(planets[ii]['magnitude_L'])-1][1]-star_mag))
			xPos = int(sz//2+100*(planets[ii]['semimajor']/dist)*np.cos(planets[ii]['pa']*np.pi/180))
			yPos = int(sz//2+100*(planets[ii]['semimajor']/dist)*np.sin(planets[ii]['pa']*np.pi/180))
			star_sim[yPos,xPos] = contrast
		#aperture = ot.circle(sz,20)
		PSF = np.abs(np.fft.fftshift(np.fft.fft2(aperture)))
		image = twoD_conv(star_sim,PSF)
		crat = cross_corr(PSF,image-PSF)/(np.sum(PSF**2))
		hdu = pyfits.PrimaryHDU(crat)
		#Create fits table with planet parameters
		cols = []
		keys = ['semimajor','pa','magnitude_L','magnitude_K','radius','current_mass','final_mass','period','age','start_date','completion_date','complete','final_entropy']
		for key in keys:
			array = []
			for ii in range(len(planets)):
				if 'magnitude' in key:
					array.append(planets[ii][key][len(planets[ii][key])-1][1])
				else:
					array.append(planets[ii][key])
			if type(array[0])==str:
				form = 'A40'
			elif type(array[0])==bool:
				form = 'B'
			else:
				form = 'E'
			if not type(array[0])==list:
				cols.append(pyfits.Column(name=key, format=form, array=array))
		cols.append(pyfits.Column(name='system_age', format='E', array=(time/1e6)*np.ones(len(planets))))
		hdu2 = pyfits.BinTableHDU.from_columns(pyfits.ColDefs(cols))
		listHDU = [hdu,hdu2]
		for ii in range(len(planets)):
			cols = []
			keys = ['magnitude_K','magnitude_L']
			tK = []
			tL = []
			Kmag = []
			Lmag = []
			if planets[ii]['completion_date']=='N/A':
				print('Incomplete: ',planets[ii]['current_mass'],planets[ii]['final_mass'])
			for key in keys:
				for jj in range(len(planets[ii][key])):
					if 'K' in key:
						tK.append(planets[ii][key][jj][0])
						Kmag.append(planets[ii][key][jj][1])
					elif 'L' in key:
                        tL.append(planets[ii][key][jj][0])
                        Lmag.append(planets[ii][key][jj][1])
			cols.append(pyfits.Column(name='magTime', format='E', array=np.array(tK)))
			cols.append(pyfits.Column(name='K_Magnitude', format='E', array=np.array(Kmag)))
			cols.append(pyfits.Column(name='L_Magnitude', format='E', array=np.array(Lmag)))
			listHDU.append(pyfits.BinTableHDU.from_columns(pyfits.ColDefs(cols)))
		hdulist = pyfits.HDUList(listHDU)
		#Write to fits file
		hdulist.writeto(save_dir+'/'+str(nSystems)+'systems_no_'+'%06d'%((nRun+1),)+'_age'+'%.3f'%(time/1e6)+'.fits',clobber=True)
	mins = []
	if len(planets)==0:
			continue
