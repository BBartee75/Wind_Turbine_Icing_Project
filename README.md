# Wind_Turbine_Icing_Project
[Python code](Possible_WTG_Icing.py) to find possible blade icing based off of temps and power output imbalance

## The Overall Goal
To identify potential icing conditions for the wind turbines, I employed a multi-criteria approach that considers both operational states: running and stopped. For turbines in operation, we examine three key factors: ambient temperature, blade pitch angle, and power output. The ambient temperature is crucial, ranging between 3°C and -17°C, indicating potential freezing conditions. I also considered the average blade angle ('Blds_PitchAngle_Avg'), looking for values less than 25 degrees, which suggests the blades are positioned for normal operation rather than extreme weather protection. Additionally, I analyzed the power output, focusing on instances where it falls below 15% of the rated power derived from the OEM power curve, categorized by wind bin increments of 0.5 m/s. This comprehensive approach helps pinpoint situations where turbines might be vulnerable to icing, even when operating normally.

Additional Technical Details
- The filtering process utilizes SCADA (Supervisory Control and Data Acquisition) parameters to ensure accurate data analysis.
- The [power curve](V100_PC.xlsx) analysis is based on wind bin divisions, allowing for precise categorization of wind speeds.
- The temperature range considered (3°C to -17°C) covers typical freezing conditions that could lead to icing. <sup>4.1 , 4.2</sup>
- The blade pitch angle threshold of 25 degrees is chosen as a balance between normal operation and extreme weather protection.
- The power output threshold of 15% of rated power is selected to identify situations where turbines might be underperforming due to potential icing effects. <sup>4.3 , 4.4</sup> 
- This approach provides a robust method for identifying potential icing conditions across various operational scenarios, enhancing the reliability and safety of wind turbine operations.

![Parameters Table.png](https://github.com/BBartee75/Wind_Turbine_Icing_Project/blob/main/Parameters%20Table.jpg)

## References
- 4.1  Scholarly journal article published in ScienceDirect: [“Phases of icing on wind turbine blades characterized by ice accumulation”](https://www.sciencedirect.com/science/article/abs/pii/S096014810900408X)
- 4.2 Scholarly journal article published in ScienceDirect: [“An experimental study on the aerodynamic performance degradation of a wind turbine blade model induced by ice accretion process”](https://www.sciencedirect.com/science/article/abs/pii/S0960148118312163)
- 4.3  Academic article published on The Conversation website: [“The science behind frozen wind turbines – and how to keep them spinning through the winter”](https://theconversation.com/the-science-behind-frozen-wind-turbines-and-how-to-keep-them-spinning-through-the-winter-156520)
- 4.4  Scholarly journal article published in ScienceDirect: [“A field study of ice accretion and its effects on the power production of utility-scale wind turbines”](https://www.sciencedirect.com/science/article/abs/pii/S0960148120319406?via%3Dihub)
