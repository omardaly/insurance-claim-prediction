### freMTPL2freq.csv ###
IDpol (int): Policy ID number; uniquely identifies each row in freMTPL2freq.csv.
ClaimNb (int): Total number of TPL motor claims made against the policy.
Exposure (float): Duration (in years) the policy was observed.
VehPower (int): Vehicle power encoded as an ordinal integer (higher values indicate more powerful vehicles).
VehAge (int): Age of the vehicle in years (integer part only).
DrivAge (int): Driver's age in years (integer part only). In France, the legal driving age is 18. The minimum driver age in this dataset is 18 years old.
BonusMalus (int): Bonus-Malus factor that ranges from 50 to 350. In France, < 100 means bonus, whilst > 100 means malus.
VehBrand (string): Anonymized vehicle brand code.
VehGas (string): Fuel type, either 'Regular' or 'Diesel'.
Area (string): Population density category of the city/community the driver lives in; these categories are ordered from "A" (rural) to "F" (urban center).
Density (int): Number of inhabitants per square-kilometer living in the city/area where the vehicle driver resides.
Region (string): Region in France the vehicle driver resides; these categories correspond to the 1982 - 2015 French region


### freMTPL2sev.csv ###
IDpol (int): Policy ID number linking the claim to a policy in freMTPL2freq.csv. Note that IDpol is not a unique row identifier in this file, as multiple claims may be made against the same policy.
ClaimAmount (float): Most up-to-date estimate of the total financial loss/cost of the claim, presumably in Euros (although the exact currency the financial losses are recorded is not disclosed)