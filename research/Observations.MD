# Observations


## Completed Chants vs Not Completed Chants
How much do affect uncompleted chants (without --3 at the end of volpiano) the results.

1. **Scores of dataset including not completed chants**

|FIELD1|genre     |subset|train_accuracy     |test_accuracy     |train_weighted_f1  |train_weighted_precision|train_weighted_recall|test_weighted_f1   |test_weighted_precision|test_weighted_recall|feature_set          |
|------|----------|------|-------------------|------------------|-------------------|------------------------|---------------------|-------------------|-----------------------|--------------------|---------------------|
|0     |responsory|full  |53.86|55.67|39.15|48.42|53.86|40.88|40.95|55.67|final                |
|1     |responsory|full  |60.81|57.75|57.86|60.89|60.81|54.94|54.56|57.75|ambitus              |
|2     |responsory|full  |47.14|46.51|38.16|41.32|47.14|37.71|41.41|46.51|initial              |
|3     |responsory|full  |90.45|88.57|90.45|90.70|90.45|88.57|88.79|88.57|final-ambitus        |
|4     |responsory|full  |75.62|75.25|74.21|79.50|75.62|73.87|79.39|75.25|final-initial        |
|5     |responsory|full  |91.95|89.62|91.95|92.06|91.95|89.58|89.69|89.62|final-ambitus-initial|
|6     |responsory|full  |73.71|70.41|73.01|75.12|73.71|69.64|69.37|70.41|ambitus-initial      |
|7     |responsory|subset|54.13|54.25|39.88|54.28|54.13|40.04|44.59|54.25|final                |
|8     |responsory|subset|57.21|49.72|55.17|59.62|57.21|47.62|46.78|49.72|ambitus              |
|9     |responsory|subset|46.03|47.07|38.57|41.61|46.03|39.36|40.21|47.07|initial              |
|10    |responsory|subset|87.84|81.10|87.78|87.92|87.84|80.95|81.27|81.10|final-ambitus        |
|11    |responsory|subset|70.18|72.59|67.70|74.63|70.18|70.44|75.10|72.59|final-initial        |
|12    |responsory|subset|91.00|81.66|90.95|91.20|91.00|81.39|81.74|81.66|final-ambitus-initial|
|13    |responsory|subset|72.37|60.87|71.89|72.80|72.37|59.84|61.26|60.87|ambitus-initial      |
|14    |antiphon  |full  |55.65|53.91|48.94|46.61|55.65|47.60|45.15|53.91|final                |
|15    |antiphon  |full  |65.66|65.79|61.64|64.93|65.66|61.83|67.36|65.79|ambitus              |
|16    |antiphon  |full  |54.75|53.81|46.28|56.60|54.75|45.49|55.37|53.81|initial              |
|17    |antiphon  |full  |81.78|79.01|81.73|82.21|81.78|78.93|79.46|79.01|final-ambitus        |
|18    |antiphon  |full  |74.86|73.05|72.93|74.13|74.86|71.22|72.74|73.05|final-initial        |
|19    |antiphon  |full  |89.30|85.57|89.25|89.52|89.30|85.44|85.82|85.57|final-ambitus-initial|
|20    |antiphon  |full  |76.03|74.27|75.12|76.20|76.03|72.87|74.29|74.27|ambitus-initial      |
|21    |antiphon  |subset|52.80|53.85|46.66|45.86|52.80|47.56|45.73|53.85|final                |
|22    |antiphon  |subset|61.35|58.33|57.96|62.65|61.35|55.07|60.89|58.33|ambitus              |
|23    |antiphon  |subset|49.67|49.36|43.06|52.92|49.67|42.55|48.77|49.36|initial              |
|24    |antiphon  |subset|79.70|75.40|79.46|80.26|79.70|75.32|75.89|75.40|final-ambitus        |
|25    |antiphon  |subset|71.66|70.43|70.93|72.53|71.66|69.89|70.92|70.43|final-initial        |
|26    |antiphon  |subset|87.87|81.89|87.76|88.09|87.87|81.76|82.48|81.89|final-ambitus-initial|
|27    |antiphon  |subset|74.03|68.51|73.85|74.63|74.03|68.36|69.15|68.51|ambitus-initial      |

2. **Scores of dataset without completed chants**
   
|FIELD1|genre     |subset|train_accuracy     |test_accuracy     |train_weighted_f1  |train_weighted_precision|train_weighted_recall|test_weighted_f1   |test_weighted_precision|test_weighted_recall|feature_set          |
|------|----------|------|-------------------|------------------|-------------------|------------------------|---------------------|-------------------|-----------------------|--------------------|---------------------|
|0     |responsory|full  |50.00|46.88|39.18|35.12|50.00|35.46|30.00|46.88|final                |
|1     |responsory|full  |64.86|59.38|62.87|71.71|64.86|53.60|59.44|59.38|ambitus              |
|2     |responsory|full  |47.30|31.25|39.61|42.28|47.30|24.74|22.47|31.25|initial              |
|3     |responsory|full  |86.49|75.00|86.33|89.96|86.49|75.25|83.83|75.00|final-ambitus        |
|4     |responsory|full  |83.78|65.62|83.57|85.01|83.78|63.51|63.47|65.62|final-initial        |
|5     |responsory|full  |94.59|68.75|94.50|95.08|94.59|67.67|71.28|68.75|final-ambitus-initial|
|6     |responsory|full  |85.14|68.75|85.17|86.13|85.14|67.62|71.03|68.75|ambitus-initial      |
|7     |responsory|subset|52.11|33.33|43.35|49.98|52.11|22.46|17.39|33.33|final                |
|8     |responsory|subset|63.38|43.33|60.00|59.47|63.38|40.11|42.01|43.33|ambitus              |
|9     |responsory|subset|47.89|46.67|40.30|39.44|47.89|36.24|31.80|46.67|initial              |
|10    |responsory|subset|87.32|73.33|87.22|87.94|87.32|71.46|71.77|73.33|final-ambitus        |
|11    |responsory|subset|77.46|66.67|76.67|82.00|77.46|63.60|65.74|66.67|final-initial        |
|12    |responsory|subset|92.96|73.33|92.92|93.65|92.96|69.88|69.77|73.33|final-ambitus-initial|
|13    |responsory|subset|84.51|73.33|84.31|86.48|84.51|72.83|74.32|73.33|ambitus-initial      |
|14    |antiphon  |full  |53.57|51.49|46.30|44.45|53.57|44.09|42.27|51.49|final                |
|15    |antiphon  |full  |65.95|66.22|62.02|65.17|65.95|62.26|63.82|66.22|ambitus              |
|16    |antiphon  |full  |54.75|53.50|48.68|60.31|54.75|47.31|59.40|53.50|initial              |
|17    |antiphon  |full  |81.13|79.34|81.13|81.92|81.13|79.35|79.87|79.34|final-ambitus        |
|18    |antiphon  |full  |76.44|74.24|75.44|76.47|76.44|72.84|74.45|74.24|final-initial        |
|19    |antiphon  |full  |89.71|86.82|89.64|89.90|89.71|86.78|86.91|86.82|final-ambitus-initial|
|20    |antiphon  |full  |76.25|74.76|75.47|76.52|76.25|74.02|75.15|74.76|ambitus-initial      |
|21    |antiphon  |subset|52.20|50.89|42.55|38.93|52.20|40.65|37.33|50.89|final                |
|22    |antiphon  |subset|61.51|61.35|58.22|62.19|61.51|58.02|60.60|61.35|ambitus              |
|23    |antiphon  |subset|51.60|47.87|45.25|55.15|51.60|41.85|49.25|47.87|initial              |
|24    |antiphon  |subset|79.14|76.51|79.07|79.14|79.14|76.53|76.79|76.51|final-ambitus        |
|25    |antiphon  |subset|75.04|71.37|74.79|75.32|75.04|70.86|71.42|71.37|final-initial        |
|26    |antiphon  |subset|88.45|84.49|88.45|88.50|88.45|84.46|84.50|84.49|final-ambitus-initial|
|27    |antiphon  |subset|74.24|69.24|74.16|75.19|74.24|69.02|70.47|69.24|ambitus-initial      |

## Overlapping N-grams using same models as bacor_ISMIR2020

|FIELD1|genre     |subset|train_accuracy     |test_accuracy      |train_weighted_f1  |train_weighted_precision|train_weighted_recall|test_weighted_f1    |test_weighted_precision|test_weighted_recall|classifier|representation      |segmentation|
|------|----------|------|-------------------|-------------------|-------------------|------------------------|---------------------|--------------------|-----------------------|--------------------|----------|--------------------|------------|
|0     |responsory|full  |94.84|92.03|94.83|94.86|94.84|91.99|92.02|92.03|linear_svc|pitch               |neumes      |
|1     |responsory|full  |96.24|92.98|96.24|96.25|96.24|92.96|93.01|92.98|linear_svc|pitch               |syllables   |
|2     |responsory|full  |97.03|91.37|97.03|97.05|97.03|91.33|91.40|91.37|linear_svc|pitch               |words       |
|3     |responsory|full  |87.36|86.39|87.32|87.37|87.36|86.39|86.46|86.39|linear_svc|pitch               |1-mer       |
|4     |responsory|full  |92.18|90.52|92.15|92.18|92.18|90.46|90.50|90.52|linear_svc|pitch               |2-mer       |
|5     |responsory|full  |94.49|92.08|94.48|94.50|94.49|92.04|92.07|92.08|linear_svc|pitch               |3-mer       |
|6     |responsory|full  |97.52|94.12|97.52|97.53|97.52|94.10|94.11|94.12|linear_svc|pitch               |4-mer       |
|7     |responsory|full  |98.84|94.12|98.84|98.84|98.84|94.11|94.13|94.12|linear_svc|pitch               |5-mer       |
|8     |responsory|full  |98.05|94.07|98.05|98.05|98.05|94.07|94.11|94.07|linear_svc|pitch               |6-mer       |
|9     |responsory|full  |96.79|90.80|96.78|96.81|96.79|90.78|90.86|90.80|linear_svc|pitch               |7-mer       |
|10    |responsory|full  |99.17|91.37|99.17|99.20|99.17|91.35|91.38|91.37|linear_svc|pitch               |8-mer       |
|11    |responsory|full  |93.72|87.01|93.77|94.14|93.72|86.98|87.47|87.01|linear_svc|pitch               |9-mer       |
|12    |responsory|full  |91.12|84.78|91.34|92.63|91.12|84.93|86.20|84.78|linear_svc|pitch               |10-mer      |
|13    |responsory|full  |85.90|81.41|86.35|89.04|85.90|81.78|84.07|81.41|linear_svc|pitch               |11-mer      |
|14    |responsory|full  |84.60|79.61|85.39|89.17|84.60|80.29|83.65|79.61|linear_svc|pitch               |12-mer      |
|15    |responsory|full  |81.31|77.10|82.29|87.11|81.31|78.00|82.40|77.10|linear_svc|pitch               |13-mer      |
|16    |responsory|full  |80.92|75.87|82.17|88.31|80.92|76.98|82.53|75.87|linear_svc|pitch               |14-mer      |
|17    |responsory|full  |77.49|72.45|78.95|87.02|77.49|73.81|81.84|72.45|linear_svc|pitch               |15-mer      |
|18    |responsory|full  |75.88|69.32|77.58|87.65|75.88|70.93|81.46|69.32|linear_svc|pitch               |16-mer      |
|19    |responsory|full  |97.05|86.06|97.05|97.07|97.05|85.82|85.95|86.06|linear_svc|pitch               |poisson-3   |
|20    |responsory|full  |99.67|78.47|99.67|99.68|99.67|78.14|78.28|78.47|linear_svc|pitch               |poisson-5   |
|21    |responsory|full  |94.03|70.13|93.98|94.15|94.03|69.11|69.58|70.13|linear_svc|pitch               |poisson-7   |
|22    |responsory|full  |98.70|86.25|98.70|98.70|98.70|86.17|86.19|86.25|linear_svc|interval-dependent  |neumes      |
|23    |responsory|full  |98.86|89.00|98.86|98.87|98.86|88.90|88.94|89.00|linear_svc|interval-dependent  |syllables   |
|24    |responsory|full  |96.71|86.53|96.70|96.75|96.71|86.34|86.71|86.53|linear_svc|interval-dependent  |words       |
|25    |responsory|full  |57.19|54.53|54.60|55.49|57.19|52.55|54.05|54.53|linear_svc|interval-dependent  |1-mer       |
|26    |responsory|full  |57.21|54.34|54.64|55.57|57.21|52.30|53.74|54.34|linear_svc|interval-dependent  |2-mer       |
|27    |responsory|full  |88.54|81.32|88.49|88.57|88.54|81.31|81.41|81.32|linear_svc|interval-dependent  |3-mer       |
|28    |responsory|full  |95.79|88.86|95.79|95.80|95.79|88.86|88.89|88.86|linear_svc|interval-dependent  |4-mer       |
|29    |responsory|full  |98.78|91.94|98.78|98.78|98.78|91.92|91.94|91.94|linear_svc|interval-dependent  |5-mer       |
|30    |responsory|full  |99.45|92.70|99.45|99.45|99.45|92.68|92.69|92.70|linear_svc|interval-dependent  |6-mer       |
|31    |responsory|full  |98.94|91.42|98.94|98.95|98.94|91.38|91.48|91.42|linear_svc|interval-dependent  |7-mer       |
|32    |responsory|full  |97.93|89.19|97.93|97.95|97.93|89.11|89.17|89.19|linear_svc|interval-dependent  |8-mer       |
|33    |responsory|full  |97.70|87.06|97.71|97.76|97.70|87.01|87.14|87.06|linear_svc|interval-dependent  |9-mer       |
|34    |responsory|full  |92.32|83.17|92.37|92.92|92.32|83.17|83.95|83.17|linear_svc|interval-dependent  |10-mer      |
|35    |responsory|full  |88.79|81.18|89.05|90.75|88.79|81.34|83.05|81.18|linear_svc|interval-dependent  |11-mer      |
|36    |responsory|full  |86.79|79.66|87.37|90.35|86.79|80.03|82.70|79.66|linear_svc|interval-dependent  |12-mer      |
|37    |responsory|full  |84.11|77.29|85.00|89.08|84.11|77.90|81.37|77.29|linear_svc|interval-dependent  |13-mer      |
|38    |responsory|full  |79.28|74.25|80.19|85.58|79.28|74.84|79.67|74.25|linear_svc|interval-dependent  |14-mer      |
|39    |responsory|full  |78.14|72.59|79.33|86.24|78.14|73.59|80.28|72.59|linear_svc|interval-dependent  |15-mer      |
|40    |responsory|full  |77.04|69.23|78.59|87.36|77.04|70.46|79.40|69.23|linear_svc|interval-dependent  |16-mer      |
|41    |responsory|full  |94.23|69.46|94.19|94.35|94.23|68.23|69.28|69.46|linear_svc|interval-dependent  |poisson-3   |
|42    |responsory|full  |96.63|63.39|96.62|96.68|96.63|61.83|62.66|63.39|linear_svc|interval-dependent  |poisson-5   |
|43    |responsory|full  |96.18|57.18|96.18|96.26|96.18|55.60|56.03|57.18|linear_svc|interval-dependent  |poisson-7   |
|44    |responsory|full  |91.89|79.33|91.82|91.94|91.89|79.01|79.20|79.33|linear_svc|interval-independent|neumes      |
|45    |responsory|full  |95.79|85.63|95.78|95.84|95.79|85.34|85.86|85.63|linear_svc|interval-independent|syllables   |
|46    |responsory|full  |98.33|86.53|98.33|98.34|98.33|86.38|86.52|86.53|linear_svc|interval-independent|words       |
|47    |responsory|full  |20.93|20.96|7.24|4.38|20.93|7.26|4.39|20.96|linear_svc|interval-independent|1-mer       |
|48    |responsory|full  |57.21|54.39|54.64|55.57|57.21|52.34|53.79|54.39|linear_svc|interval-independent|2-mer       |
|49    |responsory|full  |86.90|80.09|86.84|86.96|86.90|80.07|80.20|80.09|linear_svc|interval-independent|3-mer       |
|50    |responsory|full  |94.58|88.43|94.57|94.58|94.58|88.42|88.44|88.43|linear_svc|interval-independent|4-mer       |
|51    |responsory|full  |98.90|91.84|98.90|98.91|98.90|91.83|91.85|91.84|linear_svc|interval-independent|5-mer       |
|52    |responsory|full  |96.93|90.80|96.92|96.96|96.93|90.73|90.85|90.80|linear_svc|interval-independent|6-mer       |
|53    |responsory|full  |99.29|91.51|99.29|99.29|99.29|91.50|91.54|91.51|linear_svc|interval-independent|7-mer       |
|54    |responsory|full  |96.73|87.81|96.73|96.78|96.73|87.70|87.91|87.81|linear_svc|interval-independent|8-mer       |
|55    |responsory|full  |96.28|86.11|96.29|96.38|96.28|86.02|86.21|86.11|linear_svc|interval-independent|9-mer       |
|56    |responsory|full  |92.67|83.45|92.73|93.24|92.67|83.47|84.18|83.45|linear_svc|interval-independent|10-mer      |
|57    |responsory|full  |91.30|82.31|91.61|93.10|91.30|82.48|84.01|82.31|linear_svc|interval-independent|11-mer      |
|58    |responsory|full  |85.82|79.52|86.35|89.26|85.82|79.89|82.63|79.52|linear_svc|interval-independent|12-mer      |
|59    |responsory|full  |82.89|77.19|83.66|87.68|82.89|77.78|81.27|77.19|linear_svc|interval-independent|13-mer      |
|60    |responsory|full  |78.16|73.35|78.99|84.34|78.16|73.83|78.70|73.35|linear_svc|interval-independent|14-mer      |
|61    |responsory|full  |77.33|72.07|78.43|85.13|77.33|72.88|79.51|72.07|linear_svc|interval-independent|15-mer      |
|62    |responsory|full  |74.77|68.42|76.10|84.99|74.77|69.43|78.81|68.42|linear_svc|interval-independent|16-mer      |
|63    |responsory|full  |93.60|60.36|93.59|93.71|93.60|59.22|59.86|60.36|linear_svc|interval-independent|poisson-3   |
|64    |responsory|full  |93.90|60.46|93.86|94.05|93.90|58.40|59.38|60.46|linear_svc|interval-independent|poisson-5   |
|65    |responsory|full  |93.52|56.42|93.48|93.76|93.52|54.51|56.49|56.42|linear_svc|interval-independent|poisson-7   |
|66    |responsory|full  |73.97|62.73|73.48|74.45|73.97|61.94|62.78|62.73|linear_svc|contour-dependent   |neumes      |
|67    |responsory|full  |94.49|78.81|94.49|94.57|94.49|78.59|79.15|78.81|linear_svc|contour-dependent   |syllables   |
|68    |responsory|full  |98.76|82.93|98.76|98.76|98.76|82.81|83.04|82.93|linear_svc|contour-dependent   |words       |
|69    |responsory|full  |27.65|27.12|20.30|17.69|27.65|19.50|17.02|27.12|linear_svc|contour-dependent   |1-mer       |
|70    |responsory|full  |27.61|27.03|20.32|17.64|27.61|19.59|17.25|27.03|linear_svc|contour-dependent   |2-mer       |
|71    |responsory|full  |37.48|36.75|33.61|34.75|37.48|33.01|34.87|36.75|linear_svc|contour-dependent   |3-mer       |
|72    |responsory|full  |63.53|60.08|62.52|63.33|63.53|59.27|59.94|60.08|linear_svc|contour-dependent   |4-mer       |
|73    |responsory|full  |79.13|71.36|78.99|79.12|79.13|71.19|71.33|71.36|linear_svc|contour-dependent   |5-mer       |
|74    |responsory|full  |88.83|77.34|88.76|88.89|88.83|77.18|77.50|77.34|linear_svc|contour-dependent   |6-mer       |
|75    |responsory|full  |88.26|79.33|88.09|88.79|88.26|78.84|79.81|79.33|linear_svc|contour-dependent   |7-mer       |
|76    |responsory|full  |98.78|86.25|98.78|98.79|98.78|86.13|86.34|86.25|linear_svc|contour-dependent   |8-mer       |
|77    |responsory|full  |99.07|86.96|99.07|99.07|99.07|86.89|87.08|86.96|linear_svc|contour-dependent   |9-mer       |
|78    |responsory|full  |98.94|85.68|98.94|98.95|98.94|85.62|85.84|85.68|linear_svc|contour-dependent   |10-mer      |
|79    |responsory|full  |99.00|84.68|99.00|99.01|99.00|84.63|84.95|84.68|linear_svc|contour-dependent   |11-mer      |
|80    |responsory|full  |98.44|82.41|98.44|98.44|98.44|82.34|82.73|82.41|linear_svc|contour-dependent   |12-mer      |
|81    |responsory|full  |93.97|80.46|93.97|94.26|93.97|80.26|81.14|80.46|linear_svc|contour-dependent   |13-mer      |
|82    |responsory|full  |89.25|77.52|89.29|90.34|89.25|77.39|78.99|77.52|linear_svc|contour-dependent   |14-mer      |
|83    |responsory|full  |91.73|77.43|92.15|93.97|91.73|77.79|79.47|77.43|linear_svc|contour-dependent   |15-mer      |
|84    |responsory|full  |85.60|74.68|86.47|90.46|85.60|75.27|79.18|74.68|linear_svc|contour-dependent   |16-mer      |
|85    |responsory|full  |60.46|36.18|58.61|64.39|60.46|34.42|36.30|36.18|linear_svc|contour-dependent   |poisson-3   |
|86    |responsory|full  |79.83|40.78|79.15|81.90|79.83|38.39|39.92|40.78|linear_svc|contour-dependent   |poisson-5   |
|87    |responsory|full  |92.10|43.39|92.02|92.50|92.10|41.21|42.25|43.39|linear_svc|contour-dependent   |poisson-7   |
|88    |responsory|full  |62.74|52.73|62.18|63.53|62.74|51.85|52.48|52.73|linear_svc|contour-independent |neumes      |
|89    |responsory|full  |83.28|73.49|82.97|84.12|83.28|72.85|74.75|73.49|linear_svc|contour-independent |syllables   |
|90    |responsory|full  |98.76|81.93|98.76|98.76|98.76|81.79|82.16|81.93|linear_svc|contour-independent |words       |
|91    |responsory|full  |20.93|20.96|7.24|4.38|20.93|7.26|4.39|20.96|linear_svc|contour-independent |1-mer       |
|92    |responsory|full  |27.59|27.03|20.31|17.61|27.59|19.59|17.25|27.03|linear_svc|contour-independent |2-mer       |
|93    |responsory|full  |35.55|33.76|31.39|33.05|35.55|29.56|31.24|33.76|linear_svc|contour-independent |3-mer       |
|94    |responsory|full  |52.01|51.26|49.76|50.06|52.01|49.05|49.21|51.26|linear_svc|contour-independent |4-mer       |
|95    |responsory|full  |64.36|61.74|63.71|64.60|64.36|61.23|61.54|61.74|linear_svc|contour-independent |5-mer       |
|96    |responsory|full  |80.27|70.27|80.21|80.34|80.27|70.25|70.35|70.27|linear_svc|contour-independent |6-mer       |
|97    |responsory|full  |79.78|71.98|79.35|80.52|79.78|71.36|72.26|71.98|linear_svc|contour-independent |7-mer       |
|98    |responsory|full  |86.77|77.15|86.62|87.51|86.77|76.71|78.01|77.15|linear_svc|contour-independent |8-mer       |
|99    |responsory|full  |95.90|83.07|95.91|96.01|95.90|82.92|83.52|83.07|linear_svc|contour-independent |9-mer       |
|100   |responsory|full  |99.90|84.68|99.90|99.90|99.90|84.67|84.83|84.68|linear_svc|contour-independent |10-mer      |
|101   |responsory|full  |98.31|82.41|98.32|98.33|98.31|82.25|82.78|82.41|linear_svc|contour-independent |11-mer      |
|102   |responsory|full  |96.79|80.32|96.79|96.84|96.79|80.10|81.02|80.32|linear_svc|contour-independent |12-mer      |
|103   |responsory|full  |96.20|80.37|96.20|96.29|96.20|80.22|80.91|80.37|linear_svc|contour-independent |13-mer      |
|104   |responsory|full  |96.06|79.28|96.13|96.49|96.06|79.20|79.67|79.28|linear_svc|contour-independent |14-mer      |
|105   |responsory|full  |87.99|75.30|88.17|89.75|87.99|75.28|77.17|75.30|linear_svc|contour-independent |15-mer      |
|106   |responsory|full  |83.97|73.30|84.50|87.70|83.97|73.60|77.36|73.30|linear_svc|contour-independent |16-mer      |
|107   |responsory|full  |42.28|28.02|39.06|44.75|42.28|25.06|25.81|28.02|linear_svc|contour-independent |poisson-3   |
|108   |responsory|full  |69.97|36.04|68.79|73.75|69.97|33.40|33.21|36.04|linear_svc|contour-independent |poisson-5   |
|109   |responsory|full  |87.06|39.02|86.84|88.16|87.06|36.83|36.84|39.02|linear_svc|contour-independent |poisson-7   |
|110   |responsory|subset|94.73|82.04|94.73|94.78|94.73|81.66|81.88|82.04|linear_svc|pitch               |neumes      |
|111   |responsory|subset|94.65|84.12|94.65|94.69|94.65|83.96|84.32|84.12|linear_svc|pitch               |syllables   |
|112   |responsory|subset|99.84|70.51|99.84|99.84|99.84|69.93|70.92|70.51|linear_svc|pitch               |words       |
|113   |responsory|subset|84.36|79.02|84.34|84.39|84.36|78.83|79.45|79.02|linear_svc|pitch               |1-mer       |
|114   |responsory|subset|91.57|83.36|91.56|91.58|91.57|83.15|83.67|83.36|linear_svc|pitch               |2-mer       |
|115   |responsory|subset|94.73|85.07|94.73|94.76|94.73|84.92|85.15|85.07|linear_svc|pitch               |3-mer       |
|116   |responsory|subset|99.27|83.18|99.27|99.27|99.27|82.78|83.09|83.18|linear_svc|pitch               |4-mer       |
|117   |responsory|subset|99.35|82.80|99.35|99.36|99.35|82.51|82.75|82.80|linear_svc|pitch               |5-mer       |
|118   |responsory|subset|100.00|79.21|100.00|100.00|100.00|78.72|79.19|79.21|linear_svc|pitch               |6-mer       |
|119   |responsory|subset|98.62|77.50|98.62|98.63|98.62|76.79|77.93|77.50|linear_svc|pitch               |7-mer       |
|120   |responsory|subset|97.33|72.78|97.31|97.35|97.33|72.07|74.04|72.78|linear_svc|pitch               |8-mer       |
|121   |responsory|subset|97.08|68.24|97.11|97.26|97.08|67.58|69.45|68.24|linear_svc|pitch               |9-mer       |
|122   |responsory|subset|93.44|65.22|93.59|94.55|93.44|64.66|68.59|65.22|linear_svc|pitch               |10-mer      |
|123   |responsory|subset|87.93|62.19|88.40|91.58|87.93|61.94|68.90|62.19|linear_svc|pitch               |11-mer      |
|124   |responsory|subset|83.71|59.36|84.35|89.06|83.71|59.67|68.99|59.36|linear_svc|pitch               |12-mer      |
|125   |responsory|subset|82.25|54.63|83.26|89.41|82.25|54.56|64.31|54.63|linear_svc|pitch               |13-mer      |
|126   |responsory|subset|78.44|53.12|79.60|87.27|78.44|53.12|65.48|53.12|linear_svc|pitch               |14-mer      |
|127   |responsory|subset|75.53|50.47|76.99|87.17|75.53|49.93|63.10|50.47|linear_svc|pitch               |15-mer      |
|128   |responsory|subset|72.69|47.64|74.33|86.58|72.69|46.91|65.52|47.64|linear_svc|pitch               |16-mer      |
|129   |responsory|subset|99.84|73.16|99.84|99.84|99.84|72.39|72.99|73.16|linear_svc|pitch               |poisson-3   |
|130   |responsory|subset|100.00|62.19|100.00|100.00|100.00|61.33|62.19|62.19|linear_svc|pitch               |poisson-5   |
|131   |responsory|subset|99.84|54.63|99.84|99.84|99.84|53.22|53.56|54.63|linear_svc|pitch               |poisson-7   |
|132   |responsory|subset|100.00|67.11|100.00|100.00|100.00|66.39|66.97|67.11|linear_svc|interval-dependent  |neumes      |
|133   |responsory|subset|99.43|67.30|99.43|99.44|99.43|66.75|67.54|67.30|linear_svc|interval-dependent  |syllables   |
|134   |responsory|subset|99.84|57.84|99.84|99.84|99.84|56.48|56.81|57.84|linear_svc|interval-dependent  |words       |
|135   |responsory|subset|51.30|47.64|48.73|48.88|51.30|44.84|46.23|47.64|linear_svc|interval-dependent  |1-mer       |
|136   |responsory|subset|51.22|48.02|48.69|48.72|51.22|45.16|46.47|48.02|linear_svc|interval-dependent  |2-mer       |
|137   |responsory|subset|80.71|69.57|80.63|80.97|80.71|69.03|69.42|69.57|linear_svc|interval-dependent  |3-mer       |
|138   |responsory|subset|95.38|72.02|95.38|95.40|95.38|71.19|71.58|72.02|linear_svc|interval-dependent  |4-mer       |
|139   |responsory|subset|98.78|71.83|98.78|98.79|98.78|71.11|71.54|71.83|linear_svc|interval-dependent  |5-mer       |
|140   |responsory|subset|100.00|71.83|100.00|100.00|100.00|70.95|71.05|71.83|linear_svc|interval-dependent  |6-mer       |
|141   |responsory|subset|99.92|71.08|99.92|99.92|99.92|69.94|70.76|71.08|linear_svc|interval-dependent  |7-mer       |
|142   |responsory|subset|99.84|67.86|99.84|99.84|99.84|66.61|68.37|67.86|linear_svc|interval-dependent  |8-mer       |
|143   |responsory|subset|99.27|66.92|99.27|99.28|99.27|65.88|67.79|66.92|linear_svc|interval-dependent  |9-mer       |
|144   |responsory|subset|97.00|63.71|97.03|97.25|97.00|62.25|64.28|63.71|linear_svc|interval-dependent  |10-mer      |
|145   |responsory|subset|91.33|60.30|91.45|92.69|91.33|58.89|62.91|60.30|linear_svc|interval-dependent  |11-mer      |
|146   |responsory|subset|86.06|58.41|86.52|90.28|86.06|57.66|65.31|58.41|linear_svc|interval-dependent  |12-mer      |
|147   |responsory|subset|83.23|55.58|84.08|89.46|83.23|54.54|62.65|55.58|linear_svc|interval-dependent  |13-mer      |
|148   |responsory|subset|81.60|52.74|82.75|89.22|81.60|51.89|61.36|52.74|linear_svc|interval-dependent  |14-mer      |
|149   |responsory|subset|78.53|50.28|79.92|88.31|78.53|49.26|60.65|50.28|linear_svc|interval-dependent  |15-mer      |
|150   |responsory|subset|74.15|48.58|75.72|86.63|74.15|47.09|61.40|48.58|linear_svc|interval-dependent  |16-mer      |
|151   |responsory|subset|99.84|52.17|99.84|99.84|99.84|49.37|50.53|52.17|linear_svc|interval-dependent  |poisson-3   |
|152   |responsory|subset|99.92|44.99|99.92|99.92|99.92|42.76|46.31|44.99|linear_svc|interval-dependent  |poisson-5   |
|153   |responsory|subset|99.27|39.13|99.27|99.29|99.27|35.16|34.79|39.13|linear_svc|interval-dependent  |poisson-7   |
|154   |responsory|subset|95.62|61.81|95.62|95.65|95.62|61.30|62.21|61.81|linear_svc|interval-independent|neumes      |
|155   |responsory|subset|97.89|64.65|97.89|97.91|97.89|63.96|64.47|64.65|linear_svc|interval-independent|syllables   |
|156   |responsory|subset|99.92|59.55|99.92|99.92|99.92|58.10|58.56|59.55|linear_svc|interval-independent|words       |
|157   |responsory|subset|22.45|18.34|8.23|5.04|22.45|5.68|3.36|18.34|linear_svc|interval-independent|1-mer       |
|158   |responsory|subset|51.13|48.20|48.62|48.64|51.13|45.38|46.70|48.20|linear_svc|interval-independent|2-mer       |
|159   |responsory|subset|77.71|68.24|77.57|78.07|77.71|67.29|67.97|68.24|linear_svc|interval-independent|3-mer       |
|160   |responsory|subset|96.76|72.02|96.76|96.78|96.76|71.01|70.97|72.02|linear_svc|interval-independent|4-mer       |
|161   |responsory|subset|99.35|72.59|99.35|99.36|99.35|72.06|72.35|72.59|linear_svc|interval-independent|5-mer       |
|162   |responsory|subset|100.00|72.21|100.00|100.00|100.00|71.31|71.29|72.21|linear_svc|interval-independent|6-mer       |
|163   |responsory|subset|99.19|70.70|99.19|99.19|99.19|69.41|70.82|70.70|linear_svc|interval-independent|7-mer       |
|164   |responsory|subset|99.92|68.62|99.92|99.92|99.92|67.65|69.60|68.62|linear_svc|interval-independent|8-mer       |
|165   |responsory|subset|99.11|66.16|99.11|99.12|99.11|65.04|66.50|66.16|linear_svc|interval-independent|9-mer       |
|166   |responsory|subset|95.46|62.95|95.47|95.75|95.46|61.42|65.12|62.95|linear_svc|interval-independent|10-mer      |
|167   |responsory|subset|93.11|61.81|93.28|94.36|93.11|60.72|63.97|61.81|linear_svc|interval-independent|11-mer      |
|168   |responsory|subset|85.90|58.03|86.42|90.23|85.90|57.11|64.21|58.03|linear_svc|interval-independent|12-mer      |
|169   |responsory|subset|83.79|54.44|84.69|89.88|83.79|53.32|60.32|54.44|linear_svc|interval-independent|13-mer      |
|170   |responsory|subset|81.77|52.74|82.91|89.34|81.77|51.99|60.62|52.74|linear_svc|interval-independent|14-mer      |
|171   |responsory|subset|78.61|50.28|79.89|87.87|78.61|49.20|60.95|50.28|linear_svc|interval-independent|15-mer      |
|172   |responsory|subset|74.72|48.20|76.28|86.94|74.72|46.68|60.72|48.20|linear_svc|interval-independent|16-mer      |
|173   |responsory|subset|99.84|45.56|99.84|99.84|99.84|43.45|44.63|45.56|linear_svc|interval-independent|poisson-3   |
|174   |responsory|subset|99.92|43.10|99.92|99.92|99.92|41.25|41.35|43.10|linear_svc|interval-independent|poisson-5   |
|175   |responsory|subset|99.68|41.02|99.68|99.68|99.68|37.45|37.19|41.02|linear_svc|interval-independent|poisson-7   |
|176   |responsory|subset|62.64|51.61|60.58|66.64|62.64|48.32|50.99|51.61|linear_svc|contour-dependent   |neumes      |
|177   |responsory|subset|89.87|55.01|89.93|90.71|89.87|53.39|55.46|55.01|linear_svc|contour-dependent   |syllables   |
|178   |responsory|subset|100.00|49.53|100.00|100.00|100.00|48.54|49.81|49.53|linear_svc|contour-dependent   |words       |
|179   |responsory|subset|24.15|21.55|16.70|14.04|24.15|15.24|13.50|21.55|linear_svc|contour-dependent   |1-mer       |
|180   |responsory|subset|24.15|21.17|16.13|12.94|24.15|14.30|11.68|21.17|linear_svc|contour-dependent   |2-mer       |
|181   |responsory|subset|34.36|33.46|30.76|30.87|34.36|29.86|29.38|33.46|linear_svc|contour-dependent   |3-mer       |
|182   |responsory|subset|56.48|51.42|55.79|56.34|56.48|49.78|50.47|51.42|linear_svc|contour-dependent   |4-mer       |
|183   |responsory|subset|70.83|52.17|70.69|71.87|70.83|50.63|51.07|52.17|linear_svc|contour-dependent   |5-mer       |
|184   |responsory|subset|88.57|57.66|88.60|89.04|88.57|56.14|56.95|57.66|linear_svc|contour-dependent   |6-mer       |
|185   |responsory|subset|98.30|58.41|98.30|98.31|98.30|57.13|57.99|58.41|linear_svc|contour-dependent   |7-mer       |
|186   |responsory|subset|99.68|59.36|99.68|99.68|99.68|57.80|60.42|59.36|linear_svc|contour-dependent   |8-mer       |
|187   |responsory|subset|100.00|58.60|100.00|100.00|100.00|56.93|59.69|58.60|linear_svc|contour-dependent   |9-mer       |
|188   |responsory|subset|100.00|55.95|100.00|100.00|100.00|53.98|55.35|55.95|linear_svc|contour-dependent   |10-mer      |
|189   |responsory|subset|99.51|57.84|99.51|99.52|99.51|56.07|58.26|57.84|linear_svc|contour-dependent   |11-mer      |
|190   |responsory|subset|99.76|53.88|99.76|99.76|99.76|52.13|53.04|53.88|linear_svc|contour-dependent   |12-mer      |
|191   |responsory|subset|97.89|55.39|97.89|97.95|97.89|53.64|58.67|55.39|linear_svc|contour-dependent   |13-mer      |
|192   |responsory|subset|97.33|53.31|97.36|97.59|97.33|51.96|53.76|53.31|linear_svc|contour-dependent   |14-mer      |
|193   |responsory|subset|88.65|51.23|88.67|90.67|88.65|49.39|65.80|51.23|linear_svc|contour-dependent   |15-mer      |
|194   |responsory|subset|92.06|50.47|92.37|94.00|92.06|49.08|56.03|50.47|linear_svc|contour-dependent   |16-mer      |
|195   |responsory|subset|56.24|26.09|52.47|70.71|56.24|20.43|27.80|26.09|linear_svc|contour-dependent   |poisson-3   |
|196   |responsory|subset|97.65|28.17|97.66|97.79|97.65|25.12|29.92|28.17|linear_svc|contour-dependent   |poisson-5   |
|197   |responsory|subset|85.17|27.98|84.69|89.48|85.17|21.57|30.43|27.98|linear_svc|contour-dependent   |poisson-7   |
|198   |responsory|subset|58.10|43.10|56.66|61.09|58.10|40.79|42.42|43.10|linear_svc|contour-independent |neumes      |
|199   |responsory|subset|91.65|55.77|91.69|91.95|91.65|54.73|56.26|55.77|linear_svc|contour-independent |syllables   |
|200   |responsory|subset|100.00|47.07|100.00|100.00|100.00|46.43|47.13|47.07|linear_svc|contour-independent |words       |
|201   |responsory|subset|22.45|18.34|8.23|5.04|22.45|5.68|3.36|18.34|linear_svc|contour-independent |1-mer       |
|202   |responsory|subset|23.91|21.36|16.02|12.72|23.91|14.67|16.46|21.36|linear_svc|contour-independent |2-mer       |
|203   |responsory|subset|32.66|29.49|28.63|31.49|32.66|26.03|26.81|29.49|linear_svc|contour-independent |3-mer       |
|204   |responsory|subset|47.16|45.18|44.69|46.00|47.16|41.89|41.92|45.18|linear_svc|contour-independent |4-mer       |
|205   |responsory|subset|58.59|48.58|57.81|58.80|58.59|46.95|47.23|48.58|linear_svc|contour-independent |5-mer       |
|206   |responsory|subset|59.81|50.28|57.91|63.34|59.81|46.67|47.54|50.28|linear_svc|contour-independent |6-mer       |
|207   |responsory|subset|80.31|52.93|80.30|81.82|80.31|50.84|51.45|52.93|linear_svc|contour-independent |7-mer       |
|208   |responsory|subset|95.06|54.25|95.08|95.30|95.06|52.05|52.90|54.25|linear_svc|contour-independent |8-mer       |
|209   |responsory|subset|99.92|56.33|99.92|99.92|99.92|54.74|56.99|56.33|linear_svc|contour-independent |9-mer       |
|210   |responsory|subset|99.27|55.39|99.27|99.28|99.27|52.73|57.11|55.39|linear_svc|contour-independent |10-mer      |
|211   |responsory|subset|100.00|52.93|100.00|100.00|100.00|50.91|51.36|52.93|linear_svc|contour-independent |11-mer      |
|212   |responsory|subset|99.84|53.12|99.84|99.84|99.84|51.75|52.35|53.12|linear_svc|contour-independent |12-mer      |
|213   |responsory|subset|99.59|54.44|99.59|99.60|99.59|52.99|53.91|54.44|linear_svc|contour-independent |13-mer      |
|214   |responsory|subset|97.65|53.88|97.67|97.81|97.65|52.02|53.63|53.88|linear_svc|contour-independent |14-mer      |
|215   |responsory|subset|93.27|52.74|93.31|94.06|93.27|50.73|60.95|52.74|linear_svc|contour-independent |15-mer      |
|216   |responsory|subset|92.38|50.28|92.50|93.66|92.38|48.93|57.19|50.28|linear_svc|contour-independent |16-mer      |
|217   |responsory|subset|39.06|21.93|31.79|52.89|39.06|14.31|20.14|21.93|linear_svc|contour-independent |poisson-3   |
|218   |responsory|subset|74.31|24.57|73.60|82.51|74.31|18.71|23.99|24.57|linear_svc|contour-independent |poisson-5   |
|219   |responsory|subset|91.73|26.09|91.77|93.20|91.73|21.16|26.83|26.09|linear_svc|contour-independent |poisson-7   |
|220   |antiphon  |full  |94.15|91.97|94.11|94.16|94.15|91.91|91.98|91.97|linear_svc|pitch               |neumes      |
|221   |antiphon  |full  |94.67|92.33|94.64|94.70|94.67|92.26|92.33|92.33|linear_svc|pitch               |syllables   |
|222   |antiphon  |full  |98.59|94.69|98.58|98.60|98.59|94.63|94.70|94.69|linear_svc|pitch               |words       |
|223   |antiphon  |full  |89.37|88.63|89.28|89.39|89.37|88.51|88.64|88.63|linear_svc|pitch               |1-mer       |
|224   |antiphon  |full  |94.83|93.87|94.82|94.84|94.83|93.85|93.88|93.87|linear_svc|pitch               |2-mer       |
|225   |antiphon  |full  |97.93|95.07|97.92|97.93|97.93|95.05|95.08|95.07|linear_svc|pitch               |3-mer       |
|226   |antiphon  |full  |99.18|95.26|99.17|99.18|99.18|95.25|95.27|95.26|linear_svc|pitch               |4-mer       |
|227   |antiphon  |full  |98.75|95.72|98.75|98.75|98.75|95.69|95.75|95.72|linear_svc|pitch               |5-mer       |
|228   |antiphon  |full  |98.70|95.41|98.70|98.71|98.70|95.37|95.45|95.41|linear_svc|pitch               |6-mer       |
|229   |antiphon  |full  |99.05|94.23|99.05|99.06|99.05|94.19|94.29|94.23|linear_svc|pitch               |7-mer       |
|230   |antiphon  |full  |96.89|92.38|96.88|96.98|96.89|92.30|92.60|92.38|linear_svc|pitch               |8-mer       |
|231   |antiphon  |full  |95.09|89.35|95.11|95.55|95.09|89.27|89.98|89.35|linear_svc|pitch               |9-mer       |
|232   |antiphon  |full  |89.55|84.88|89.61|91.47|89.55|84.80|87.12|84.88|linear_svc|pitch               |10-mer      |
|233   |antiphon  |full  |84.44|80.24|84.70|89.03|84.44|80.38|85.63|80.24|linear_svc|pitch               |11-mer      |
|234   |antiphon  |full  |78.22|73.62|78.51|86.63|78.22|73.90|84.09|73.62|linear_svc|pitch               |12-mer      |
|235   |antiphon  |full  |72.93|68.38|73.38|85.35|72.93|68.83|83.35|68.38|linear_svc|pitch               |13-mer      |
|236   |antiphon  |full  |67.76|62.44|67.93|84.12|67.76|62.39|82.73|62.44|linear_svc|pitch               |14-mer      |
|237   |antiphon  |full  |62.98|56.50|62.71|83.21|62.98|55.57|82.15|56.50|linear_svc|pitch               |15-mer      |
|238   |antiphon  |full  |59.65|51.57|59.18|82.60|59.65|49.66|81.73|51.57|linear_svc|pitch               |16-mer      |
|239   |antiphon  |full  |99.04|88.29|99.04|99.04|99.04|88.16|88.27|88.29|linear_svc|pitch               |poisson-3   |
|240   |antiphon  |full  |97.72|85.28|97.72|97.73|97.72|85.05|85.34|85.28|linear_svc|pitch               |poisson-5   |
|241   |antiphon  |full  |93.88|79.68|93.83|94.06|93.88|79.13|79.82|79.68|linear_svc|pitch               |poisson-7   |
|242   |antiphon  |full  |86.43|79.85|86.38|86.61|86.43|79.51|79.97|79.85|linear_svc|interval-dependent  |neumes      |
|243   |antiphon  |full  |88.55|81.32|88.52|88.69|88.55|81.00|81.32|81.32|linear_svc|interval-dependent  |syllables   |
|244   |antiphon  |full  |99.65|92.28|99.65|99.65|99.65|92.23|92.32|92.28|linear_svc|interval-dependent  |words       |
|245   |antiphon  |full  |58.58|57.87|54.69|59.57|58.58|54.14|59.43|57.87|linear_svc|interval-dependent  |1-mer       |
|246   |antiphon  |full  |58.43|57.87|54.47|59.62|58.43|54.06|59.46|57.87|linear_svc|interval-dependent  |2-mer       |
|247   |antiphon  |full  |90.55|87.50|90.52|90.57|90.55|87.45|87.55|87.50|linear_svc|interval-dependent  |3-mer       |
|248   |antiphon  |full  |96.34|92.88|96.33|96.36|96.34|92.84|92.89|92.88|linear_svc|interval-dependent  |4-mer       |
|249   |antiphon  |full  |98.98|94.25|98.98|98.98|98.98|94.20|94.27|94.25|linear_svc|interval-dependent  |5-mer       |
|250   |antiphon  |full  |99.31|94.47|99.31|99.31|99.31|94.43|94.49|94.47|linear_svc|interval-dependent  |6-mer       |
|251   |antiphon  |full  |99.57|93.22|99.57|99.57|99.57|93.16|93.25|93.22|linear_svc|interval-dependent  |7-mer       |
|252   |antiphon  |full  |98.99|92.11|98.99|99.00|98.99|92.04|92.20|92.11|linear_svc|interval-dependent  |8-mer       |
|253   |antiphon  |full  |97.04|89.76|97.04|97.14|97.04|89.61|89.98|89.76|linear_svc|interval-dependent  |9-mer       |
|254   |antiphon  |full  |91.20|85.91|91.13|92.16|91.20|85.60|87.37|85.91|linear_svc|interval-dependent  |10-mer      |
|255   |antiphon  |full  |89.35|81.92|89.50|91.66|89.35|81.81|84.30|81.92|linear_svc|interval-dependent  |11-mer      |
|256   |antiphon  |full  |82.42|76.15|82.69|87.91|82.42|76.15|82.78|76.15|linear_svc|interval-dependent  |12-mer      |
|257   |antiphon  |full  |75.62|70.91|75.87|85.50|75.62|71.03|82.80|70.91|linear_svc|interval-dependent  |13-mer      |
|258   |antiphon  |full  |70.55|64.82|70.71|84.55|70.55|64.60|82.05|64.82|linear_svc|interval-dependent  |14-mer      |
|259   |antiphon  |full  |65.49|59.37|65.41|83.49|65.49|58.79|81.45|59.37|linear_svc|interval-dependent  |15-mer      |
|260   |antiphon  |full  |61.18|54.36|60.63|82.58|61.18|52.91|81.33|54.36|linear_svc|interval-dependent  |16-mer      |
|261   |antiphon  |full  |96.99|79.18|96.99|97.02|96.99|78.78|79.28|79.18|linear_svc|interval-dependent  |poisson-3   |
|262   |antiphon  |full  |95.79|76.32|95.76|95.84|95.79|75.66|76.63|76.32|linear_svc|interval-dependent  |poisson-5   |
|263   |antiphon  |full  |91.04|73.12|90.93|91.31|91.04|72.36|73.79|73.12|linear_svc|interval-dependent  |poisson-7   |
|264   |antiphon  |full  |58.79|51.96|56.62|61.64|58.79|48.58|51.48|51.96|linear_svc|interval-independent|neumes      |
|265   |antiphon  |full  |64.01|55.16|62.68|66.64|64.01|52.53|54.44|55.16|linear_svc|interval-independent|syllables   |
|266   |antiphon  |full  |94.83|89.47|94.78|95.01|94.83|89.31|90.03|89.47|linear_svc|interval-independent|words       |
|267   |antiphon  |full  |28.64|27.72|12.75|8.20|28.64|12.03|7.69|27.72|linear_svc|interval-independent|1-mer       |
|268   |antiphon  |full  |58.25|57.54|54.13|59.30|58.25|53.49|59.15|57.54|linear_svc|interval-independent|2-mer       |
|269   |antiphon  |full  |87.53|85.43|87.48|87.57|87.53|85.33|85.46|85.43|linear_svc|interval-independent|3-mer       |
|270   |antiphon  |full  |97.20|91.95|97.19|97.20|97.20|91.91|91.94|91.95|linear_svc|interval-independent|4-mer       |
|271   |antiphon  |full  |99.04|93.82|99.04|99.04|99.04|93.78|93.83|93.82|linear_svc|interval-independent|5-mer       |
|272   |antiphon  |full  |99.40|94.33|99.40|99.40|99.40|94.28|94.35|94.33|linear_svc|interval-independent|6-mer       |
|273   |antiphon  |full  |99.47|93.29|99.47|99.48|99.47|93.22|93.33|93.29|linear_svc|interval-independent|7-mer       |
|274   |antiphon  |full  |98.74|92.33|98.74|98.75|98.74|92.24|92.42|92.33|linear_svc|interval-independent|8-mer       |
|275   |antiphon  |full  |97.87|89.97|97.87|97.92|97.87|89.84|90.14|89.97|linear_svc|interval-independent|9-mer       |
|276   |antiphon  |full  |92.90|86.61|92.89|93.57|92.90|86.35|87.52|86.61|linear_svc|interval-independent|10-mer      |
|277   |antiphon  |full  |87.33|81.85|87.31|89.79|87.33|81.60|84.63|81.85|linear_svc|interval-independent|11-mer      |
|278   |antiphon  |full  |81.14|76.22|81.21|86.80|81.14|76.08|83.33|76.22|linear_svc|interval-independent|12-mer      |
|279   |antiphon  |full  |75.96|70.93|76.26|85.70|75.96|71.02|82.61|70.93|linear_svc|interval-independent|13-mer      |
|280   |antiphon  |full  |70.71|65.06|70.88|84.41|70.71|64.91|82.35|65.06|linear_svc|interval-independent|14-mer      |
|281   |antiphon  |full  |65.30|59.70|65.22|83.39|65.30|59.16|81.70|59.70|linear_svc|interval-independent|15-mer      |
|282   |antiphon  |full  |61.36|54.46|60.96|82.82|61.36|53.13|80.57|54.46|linear_svc|interval-independent|16-mer      |
|283   |antiphon  |full  |89.54|68.89|89.47|89.81|89.54|67.98|68.66|68.89|linear_svc|interval-independent|poisson-3   |
|284   |antiphon  |full  |95.99|72.23|95.98|96.03|95.99|71.59|72.21|72.23|linear_svc|interval-independent|poisson-5   |
|285   |antiphon  |full  |84.93|69.58|84.50|85.93|84.93|68.20|71.49|69.58|linear_svc|interval-independent|poisson-7   |
|286   |antiphon  |full  |46.34|43.14|41.94|49.91|46.34|38.48|43.70|43.14|linear_svc|contour-dependent   |neumes      |
|287   |antiphon  |full  |52.31|46.91|49.94|56.05|52.31|43.58|47.20|46.91|linear_svc|contour-dependent   |syllables   |
|288   |antiphon  |full  |94.23|85.50|94.21|94.32|94.23|85.38|85.73|85.50|linear_svc|contour-dependent   |words       |
|289   |antiphon  |full  |33.06|33.33|23.60|28.00|33.06|23.53|27.20|33.33|linear_svc|contour-dependent   |1-mer       |
|290   |antiphon  |full  |32.81|33.13|23.52|28.36|32.81|23.43|26.60|33.13|linear_svc|contour-dependent   |2-mer       |
|291   |antiphon  |full  |42.04|42.94|35.15|37.88|42.04|36.01|39.91|42.94|linear_svc|contour-dependent   |3-mer       |
|292   |antiphon  |full  |64.96|63.96|64.14|65.02|64.96|62.96|63.86|63.96|linear_svc|contour-dependent   |4-mer       |
|293   |antiphon  |full  |82.85|81.37|82.78|82.87|82.85|81.27|81.40|81.37|linear_svc|contour-dependent   |5-mer       |
|294   |antiphon  |full  |91.54|85.77|91.51|91.56|91.54|85.65|85.81|85.77|linear_svc|contour-dependent   |6-mer       |
|295   |antiphon  |full  |95.68|88.31|95.67|95.71|95.68|88.23|88.36|88.31|linear_svc|contour-dependent   |7-mer       |
|296   |antiphon  |full  |98.91|88.94|98.91|98.91|98.91|88.85|89.03|88.94|linear_svc|contour-dependent   |8-mer       |
|297   |antiphon  |full  |99.74|87.40|99.74|99.74|99.74|87.31|87.43|87.40|linear_svc|contour-dependent   |9-mer       |
|298   |antiphon  |full  |97.78|84.85|97.78|97.80|97.78|84.65|85.06|84.85|linear_svc|contour-dependent   |10-mer      |
|299   |antiphon  |full  |96.34|81.37|96.34|96.44|96.34|81.03|81.67|81.37|linear_svc|contour-dependent   |11-mer      |
|300   |antiphon  |full  |95.19|77.88|95.21|95.50|95.19|77.56|78.44|77.88|linear_svc|contour-dependent   |12-mer      |
|301   |antiphon  |full  |86.87|74.18|86.83|88.71|86.87|73.51|77.03|74.18|linear_svc|contour-dependent   |13-mer      |
|302   |antiphon  |full  |83.28|70.50|83.48|87.48|83.28|69.86|75.00|70.50|linear_svc|contour-dependent   |14-mer      |
|303   |antiphon  |full  |73.38|65.02|73.09|82.33|73.38|64.20|75.98|65.02|linear_svc|contour-dependent   |15-mer      |
|304   |antiphon  |full  |69.49|60.93|69.52|82.91|69.49|59.99|77.19|60.93|linear_svc|contour-dependent   |16-mer      |
|305   |antiphon  |full  |57.82|42.68|54.96|61.62|57.82|37.45|39.23|42.68|linear_svc|contour-dependent   |poisson-3   |
|306   |antiphon  |full  |78.09|51.55|77.46|79.81|78.09|48.27|50.16|51.55|linear_svc|contour-dependent   |poisson-5   |
|307   |antiphon  |full  |83.78|55.49|83.48|84.86|83.78|53.00|55.09|55.49|linear_svc|contour-dependent   |poisson-7   |
|308   |antiphon  |full  |39.21|36.98|31.88|45.81|39.21|29.62|40.52|36.98|linear_svc|contour-independent |neumes      |
|309   |antiphon  |full  |45.00|40.54|40.70|52.97|45.00|34.76|42.14|40.54|linear_svc|contour-independent |syllables   |
|310   |antiphon  |full  |89.58|82.90|89.52|89.87|89.58|82.75|83.50|82.90|linear_svc|contour-independent |words       |
|311   |antiphon  |full  |28.64|27.72|12.75|8.20|28.64|12.03|7.69|27.72|linear_svc|contour-independent |1-mer       |
|312   |antiphon  |full  |32.80|33.11|23.53|28.21|32.80|23.43|26.34|33.11|linear_svc|contour-independent |2-mer       |
|313   |antiphon  |full  |38.93|39.34|30.70|34.39|38.93|30.86|31.08|39.34|linear_svc|contour-independent |3-mer       |
|314   |antiphon  |full  |53.42|52.49|50.15|53.80|53.42|48.97|52.37|52.49|linear_svc|contour-independent |4-mer       |
|315   |antiphon  |full  |72.07|71.00|71.54|71.92|72.07|70.48|71.04|71.00|linear_svc|contour-independent |5-mer       |
|316   |antiphon  |full  |84.82|81.53|84.73|84.81|84.82|81.35|81.53|81.53|linear_svc|contour-independent |6-mer       |
|317   |antiphon  |full  |92.05|84.18|92.03|92.07|92.05|84.02|84.12|84.18|linear_svc|contour-independent |7-mer       |
|318   |antiphon  |full  |93.55|84.28|93.53|93.62|93.55|84.10|84.36|84.28|linear_svc|contour-independent |8-mer       |
|319   |antiphon  |full  |97.77|84.95|97.77|97.79|97.77|84.77|85.06|84.95|linear_svc|contour-independent |9-mer       |
|320   |antiphon  |full  |97.62|83.99|97.62|97.65|97.62|83.79|84.02|83.99|linear_svc|contour-independent |10-mer      |
|321   |antiphon  |full  |94.70|80.19|94.68|94.83|94.70|79.82|80.75|80.19|linear_svc|contour-independent |11-mer      |
|322   |antiphon  |full  |97.03|78.07|97.05|97.18|97.03|77.81|78.22|78.07|linear_svc|contour-independent |12-mer      |
|323   |antiphon  |full  |83.80|72.69|83.52|86.12|83.80|71.77|75.83|72.69|linear_svc|contour-independent |13-mer      |
|324   |antiphon  |full  |85.03|70.62|85.18|88.26|85.03|69.85|73.79|70.62|linear_svc|contour-independent |14-mer      |
|325   |antiphon  |full  |78.55|66.34|78.79|85.20|78.55|65.66|74.37|66.34|linear_svc|contour-independent |15-mer      |
|326   |antiphon  |full  |72.47|62.03|72.67|83.55|72.47|61.15|74.98|62.03|linear_svc|contour-independent |16-mer      |
|327   |antiphon  |full  |42.82|37.92|35.61|50.18|42.82|30.21|32.40|37.92|linear_svc|contour-independent |poisson-3   |
|328   |antiphon  |full  |65.15|44.22|63.02|69.59|65.15|39.78|44.35|44.22|linear_svc|contour-independent |poisson-5   |
|329   |antiphon  |full  |88.01|49.89|87.96|88.37|88.01|48.09|48.05|49.89|linear_svc|contour-independent |poisson-7   |
|330   |antiphon  |subset|92.85|87.98|92.82|92.87|92.85|87.90|88.05|87.98|linear_svc|pitch               |neumes      |
|331   |antiphon  |subset|94.06|87.34|94.04|94.08|94.06|87.28|87.38|87.34|linear_svc|pitch               |syllables   |
|332   |antiphon  |subset|99.38|88.38|99.38|99.39|99.38|88.29|88.47|88.38|linear_svc|pitch               |words       |
|333   |antiphon  |subset|88.25|83.89|88.19|88.26|88.25|83.84|84.03|83.89|linear_svc|pitch               |1-mer       |
|334   |antiphon  |subset|93.71|89.50|93.70|93.72|93.71|89.52|89.62|89.50|linear_svc|pitch               |2-mer       |
|335   |antiphon  |subset|94.43|91.43|94.43|94.45|94.43|91.41|91.46|91.43|linear_svc|pitch               |3-mer       |
|336   |antiphon  |subset|96.32|91.03|96.32|96.34|96.32|91.00|91.07|91.03|linear_svc|pitch               |4-mer       |
|337   |antiphon  |subset|99.42|91.51|99.42|99.42|99.42|91.47|91.54|91.51|linear_svc|pitch               |5-mer       |
|338   |antiphon  |subset|98.42|89.90|98.42|98.43|98.42|89.86|90.05|89.90|linear_svc|pitch               |6-mer       |
|339   |antiphon  |subset|99.55|87.66|99.55|99.56|99.55|87.63|87.80|87.66|linear_svc|pitch               |7-mer       |
|340   |antiphon  |subset|98.28|85.98|98.29|98.36|98.28|85.93|86.47|85.98|linear_svc|pitch               |8-mer       |
|341   |antiphon  |subset|95.98|82.29|96.04|96.40|95.98|82.23|83.66|82.29|linear_svc|pitch               |9-mer       |
|342   |antiphon  |subset|91.55|76.68|91.79|93.29|91.55|76.78|81.51|76.68|linear_svc|pitch               |10-mer      |
|343   |antiphon  |subset|87.43|70.11|88.08|91.56|87.43|70.55|79.25|70.11|linear_svc|pitch               |11-mer      |
|344   |antiphon  |subset|82.65|62.74|83.69|89.57|82.65|63.39|78.69|62.74|linear_svc|pitch               |12-mer      |
|345   |antiphon  |subset|76.71|57.13|78.28|88.03|76.71|57.91|80.19|57.13|linear_svc|pitch               |13-mer      |
|346   |antiphon  |subset|73.10|50.48|74.88|87.16|73.10|50.55|80.48|50.48|linear_svc|pitch               |14-mer      |
|347   |antiphon  |subset|74.20|42.79|75.81|87.49|74.20|40.96|79.28|42.79|linear_svc|pitch               |15-mer      |
|348   |antiphon  |subset|74.30|36.70|75.65|87.48|74.30|32.54|78.80|36.70|linear_svc|pitch               |16-mer      |
|349   |antiphon  |subset|99.66|81.17|99.66|99.66|99.66|81.06|81.15|81.17|linear_svc|pitch               |poisson-3   |
|350   |antiphon  |subset|99.38|75.16|99.38|99.39|99.38|74.53|74.88|75.16|linear_svc|pitch               |poisson-5   |
|351   |antiphon  |subset|98.32|68.11|98.32|98.35|98.32|67.53|68.22|68.11|linear_svc|pitch               |poisson-7   |
|352   |antiphon  |subset|88.63|70.91|88.61|88.81|88.63|70.39|71.04|70.91|linear_svc|interval-dependent  |neumes      |
|353   |antiphon  |subset|89.28|72.28|89.27|89.46|89.28|71.78|72.54|72.28|linear_svc|interval-dependent  |syllables   |
|354   |antiphon  |subset|98.97|82.45|98.97|98.98|98.97|82.20|82.95|82.45|linear_svc|interval-dependent  |words       |
|355   |antiphon  |subset|55.14|53.45|51.92|56.61|55.14|49.88|52.91|53.45|linear_svc|interval-dependent  |1-mer       |
|356   |antiphon  |subset|55.03|53.45|51.77|56.67|55.03|49.83|53.27|53.45|linear_svc|interval-dependent  |2-mer       |
|357   |antiphon  |subset|87.26|80.53|87.25|87.36|87.26|80.38|80.48|80.53|linear_svc|interval-dependent  |3-mer       |
|358   |antiphon  |subset|96.22|86.38|96.21|96.24|96.22|86.29|86.44|86.38|linear_svc|interval-dependent  |4-mer       |
|359   |antiphon  |subset|99.07|88.06|99.07|99.08|99.07|87.99|88.21|88.06|linear_svc|interval-dependent  |5-mer       |
|360   |antiphon  |subset|99.11|88.62|99.11|99.11|99.11|88.50|88.80|88.62|linear_svc|interval-dependent  |6-mer       |
|361   |antiphon  |subset|98.97|86.78|98.97|98.98|98.97|86.63|86.93|86.78|linear_svc|interval-dependent  |7-mer       |
|362   |antiphon  |subset|99.48|84.29|99.49|99.49|99.48|84.20|84.45|84.29|linear_svc|interval-dependent  |8-mer       |
|363   |antiphon  |subset|97.87|80.77|97.88|97.93|97.87|80.54|81.45|80.77|linear_svc|interval-dependent  |9-mer       |
|364   |antiphon  |subset|94.68|75.96|94.73|95.17|94.68|75.65|77.87|75.96|linear_svc|interval-dependent  |10-mer      |
|365   |antiphon  |subset|90.11|71.63|90.40|92.36|90.11|71.48|77.13|71.63|linear_svc|interval-dependent  |11-mer      |
|366   |antiphon  |subset|86.71|64.26|87.36|91.14|86.71|64.31|74.16|64.26|linear_svc|interval-dependent  |12-mer      |
|367   |antiphon  |subset|82.17|59.13|83.28|89.61|82.17|59.70|75.84|59.13|linear_svc|interval-dependent  |13-mer      |
|368   |antiphon  |subset|74.82|53.85|76.46|87.28|74.82|53.87|77.79|53.85|linear_svc|interval-dependent  |14-mer      |
|369   |antiphon  |subset|74.48|45.11|76.12|87.32|74.48|43.82|75.08|45.11|linear_svc|interval-dependent  |15-mer      |
|370   |antiphon  |subset|75.09|39.58|76.68|87.55|75.09|36.66|79.00|39.58|linear_svc|interval-dependent  |16-mer      |
|371   |antiphon  |subset|99.62|68.03|99.62|99.62|99.62|67.51|67.83|68.03|linear_svc|interval-dependent  |poisson-3   |
|372   |antiphon  |subset|98.76|64.98|98.76|98.77|98.76|63.66|66.10|64.98|linear_svc|interval-dependent  |poisson-5   |
|373   |antiphon  |subset|98.59|59.78|98.60|98.64|98.59|58.90|59.37|59.78|linear_svc|interval-dependent  |poisson-7   |
|374   |antiphon  |subset|55.69|45.67|53.62|60.46|55.69|42.65|45.84|45.67|linear_svc|interval-independent|neumes      |
|375   |antiphon  |subset|64.96|46.79|64.20|68.57|64.96|44.22|46.20|46.79|linear_svc|interval-independent|syllables   |
|376   |antiphon  |subset|98.56|82.13|98.56|98.57|98.56|81.95|82.43|82.13|linear_svc|interval-independent|words       |
|377   |antiphon  |subset|23.74|23.64|9.11|5.63|23.74|9.04|5.59|23.64|linear_svc|interval-independent|1-mer       |
|378   |antiphon  |subset|55.03|53.45|51.77|56.64|55.03|49.83|53.27|53.45|linear_svc|interval-independent|2-mer       |
|379   |antiphon  |subset|86.19|79.33|86.17|86.28|86.19|79.24|79.48|79.33|linear_svc|interval-independent|3-mer       |
|380   |antiphon  |subset|97.66|85.10|97.66|97.67|97.66|85.03|85.23|85.10|linear_svc|interval-independent|4-mer       |
|381   |antiphon  |subset|96.60|87.82|96.59|96.65|96.60|87.69|88.05|87.82|linear_svc|interval-independent|5-mer       |
|382   |antiphon  |subset|99.31|88.22|99.31|99.32|99.31|88.11|88.38|88.22|linear_svc|interval-independent|6-mer       |
|383   |antiphon  |subset|99.83|86.94|99.83|99.83|99.83|86.93|87.12|86.94|linear_svc|interval-independent|7-mer       |
|384   |antiphon  |subset|98.49|84.21|98.49|98.51|98.49|84.01|84.49|84.21|linear_svc|interval-independent|8-mer       |
|385   |antiphon  |subset|97.73|81.01|97.74|97.80|97.73|80.74|81.71|81.01|linear_svc|interval-independent|9-mer       |
|386   |antiphon  |subset|96.53|76.92|96.58|96.90|96.53|76.73|78.10|76.92|linear_svc|interval-independent|10-mer      |
|387   |antiphon  |subset|91.38|72.20|91.64|93.14|91.38|71.98|76.28|72.20|linear_svc|interval-independent|11-mer      |
|388   |antiphon  |subset|87.08|66.11|87.70|91.30|87.08|66.49|75.72|66.11|linear_svc|interval-independent|12-mer      |
|389   |antiphon  |subset|81.86|59.94|82.87|89.12|81.86|60.18|76.50|59.94|linear_svc|interval-independent|13-mer      |
|390   |antiphon  |subset|75.37|53.37|76.96|87.41|75.37|53.50|75.81|53.37|linear_svc|interval-independent|14-mer      |
|391   |antiphon  |subset|73.75|45.75|75.47|86.94|73.75|44.62|75.85|45.75|linear_svc|interval-independent|15-mer      |
|392   |antiphon  |subset|75.95|39.50|77.62|87.96|75.95|36.45|75.40|39.50|linear_svc|interval-independent|16-mer      |
|393   |antiphon  |subset|98.56|56.09|98.56|98.57|98.56|55.20|55.41|56.09|linear_svc|interval-independent|poisson-3   |
|394   |antiphon  |subset|99.79|58.73|99.79|99.79|99.79|57.83|58.68|58.73|linear_svc|interval-independent|poisson-5   |
|395   |antiphon  |subset|97.15|58.33|97.15|97.22|97.15|56.79|58.56|58.33|linear_svc|interval-independent|poisson-7   |
|396   |antiphon  |subset|46.14|36.70|42.57|51.80|46.14|32.40|35.45|36.70|linear_svc|contour-dependent   |neumes      |
|397   |antiphon  |subset|53.52|39.98|51.97|57.18|53.52|37.40|40.59|39.98|linear_svc|contour-dependent   |syllables   |
|398   |antiphon  |subset|98.39|76.04|98.39|98.40|98.39|75.86|76.15|76.04|linear_svc|contour-dependent   |words       |
|399   |antiphon  |subset|29.54|28.69|19.85|19.00|29.54|19.51|19.52|28.69|linear_svc|contour-dependent   |1-mer       |
|400   |antiphon  |subset|29.85|28.85|20.29|18.36|29.85|19.70|18.39|28.85|linear_svc|contour-dependent   |2-mer       |
|401   |antiphon  |subset|36.89|37.58|31.61|34.46|36.89|31.64|33.47|37.58|linear_svc|contour-dependent   |3-mer       |
|402   |antiphon  |subset|59.81|55.05|58.94|60.11|59.81|54.11|55.34|55.05|linear_svc|contour-dependent   |4-mer       |
|403   |antiphon  |subset|80.93|70.59|80.84|80.93|80.93|70.32|70.35|70.59|linear_svc|contour-dependent   |5-mer       |
|404   |antiphon  |subset|91.89|76.04|91.89|91.94|91.89|75.90|75.92|76.04|linear_svc|contour-dependent   |6-mer       |
|405   |antiphon  |subset|97.97|77.00|97.97|97.99|97.97|76.77|77.17|77.00|linear_svc|contour-dependent   |7-mer       |
|406   |antiphon  |subset|98.32|77.08|98.32|98.33|98.32|76.78|77.80|77.08|linear_svc|contour-dependent   |8-mer       |
|407   |antiphon  |subset|99.83|74.04|99.83|99.83|99.83|73.78|74.29|74.04|linear_svc|contour-dependent   |9-mer       |
|408   |antiphon  |subset|96.91|69.63|96.91|96.99|96.91|68.58|71.80|69.63|linear_svc|contour-dependent   |10-mer      |
|409   |antiphon  |subset|98.97|66.11|98.97|99.00|98.97|65.30|66.60|66.11|linear_svc|contour-dependent   |11-mer      |
|410   |antiphon  |subset|96.60|63.14|96.64|96.88|96.60|62.00|64.43|63.14|linear_svc|contour-dependent   |12-mer      |
|411   |antiphon  |subset|93.37|58.01|93.57|94.66|93.37|57.08|60.14|58.01|linear_svc|contour-dependent   |13-mer      |
|412   |antiphon  |subset|88.63|56.65|89.09|91.72|88.63|55.14|62.71|56.65|linear_svc|contour-dependent   |14-mer      |
|413   |antiphon  |subset|86.60|51.68|87.32|91.12|86.60|51.11|61.34|51.68|linear_svc|contour-dependent   |15-mer      |
|414   |antiphon  |subset|81.04|46.31|82.29|89.24|81.04|45.31|64.42|46.31|linear_svc|contour-dependent   |16-mer      |
|415   |antiphon  |subset|58.33|34.54|55.65|66.63|58.33|29.52|36.11|34.54|linear_svc|contour-dependent   |poisson-3   |
|416   |antiphon  |subset|67.23|37.90|64.40|76.57|67.23|31.10|35.34|37.90|linear_svc|contour-dependent   |poisson-5   |
|417   |antiphon  |subset|97.97|41.43|97.97|98.01|97.97|38.03|38.70|41.43|linear_svc|contour-dependent   |poisson-7   |
|418   |antiphon  |subset|35.97|32.13|29.59|46.03|35.97|25.03|34.24|32.13|linear_svc|contour-independent |neumes      |
|419   |antiphon  |subset|43.18|32.21|39.82|52.53|43.18|26.88|30.85|32.21|linear_svc|contour-independent |syllables   |
|420   |antiphon  |subset|87.01|73.88|86.97|87.75|87.01|73.72|75.99|73.88|linear_svc|contour-independent |words       |
|421   |antiphon  |subset|23.74|23.64|9.11|5.63|23.74|9.04|5.59|23.64|linear_svc|contour-independent |1-mer       |
|422   |antiphon  |subset|29.78|28.85|20.80|17.74|29.78|20.17|17.82|28.85|linear_svc|contour-independent |2-mer       |
|423   |antiphon  |subset|33.08|32.85|26.02|27.71|33.08|25.71|29.52|32.85|linear_svc|contour-independent |3-mer       |
|424   |antiphon  |subset|47.27|42.95|43.84|45.79|47.27|39.20|45.41|42.95|linear_svc|contour-independent |4-mer       |
|425   |antiphon  |subset|67.47|61.86|67.00|67.44|67.47|61.31|61.68|61.86|linear_svc|contour-independent |5-mer       |
|426   |antiphon  |subset|80.87|71.47|80.78|81.01|80.87|71.22|71.40|71.47|linear_svc|contour-independent |6-mer       |
|427   |antiphon  |subset|85.95|70.59|85.89|86.14|85.95|70.09|70.63|70.59|linear_svc|contour-independent |7-mer       |
|428   |antiphon  |subset|98.49|70.75|98.49|98.49|98.49|70.45|70.61|70.75|linear_svc|contour-independent |8-mer       |
|429   |antiphon  |subset|91.51|68.59|91.48|91.99|91.51|67.53|71.39|68.59|linear_svc|contour-independent |9-mer       |
|430   |antiphon  |subset|99.73|65.79|99.73|99.73|99.73|65.43|65.67|65.79|linear_svc|contour-independent |10-mer      |
|431   |antiphon  |subset|98.63|63.70|98.63|98.65|98.63|62.82|63.97|63.70|linear_svc|contour-independent |11-mer      |
|432   |antiphon  |subset|96.08|61.38|96.12|96.33|96.08|59.90|62.47|61.38|linear_svc|contour-independent |12-mer      |
|433   |antiphon  |subset|94.92|57.45|95.05|95.68|94.92|56.46|58.01|57.45|linear_svc|contour-independent |13-mer      |
|434   |antiphon  |subset|91.07|57.37|91.44|93.26|91.07|56.12|61.15|57.37|linear_svc|contour-independent |14-mer      |
|435   |antiphon  |subset|87.26|53.29|87.88|91.35|87.26|52.06|61.04|53.29|linear_svc|contour-independent |15-mer      |
|436   |antiphon  |subset|84.40|46.07|85.30|90.25|84.40|44.78|59.67|46.07|linear_svc|contour-independent |16-mer      |
|437   |antiphon  |subset|36.93|29.97|27.35|45.72|36.93|20.76|28.70|29.97|linear_svc|contour-independent |poisson-3   |
|438   |antiphon  |subset|71.21|33.57|69.67|76.81|71.21|28.68|34.57|33.57|linear_svc|contour-independent |poisson-5   |
|439   |antiphon  |subset|91.31|38.70|91.34|91.94|91.31|35.12|37.01|38.70|linear_svc|contour-independent |poisson-7   |
