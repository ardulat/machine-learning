import eu.verdelhan.ta4j.indicators.*;
import eu.verdelhan.ta4j.indicators.volume.ChaikinMoneyFlowIndicator;
import eu.verdelhan.ta4j.indicators.helpers.ClosePriceIndicator;

import java.io.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class build a Ta4j time series from a CSV file containing bars.
 */
public class CnnTaIndicators {

    public static void main(String[] args) {

        String[] dow_stocks = {"AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS", "GE",
                "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE",
                "PFE", "PG", "TRV", "UNH", "UTX", "VZ", "WMT", "XOM", "DWDP"};
//        String[] dow_stocks = {"AAPL"};

        for (String stockName : dow_stocks) {
            eu.verdelhan.ta4j.TimeSeries series = OldCsvLoader.loadDow30Series("Dow 30 2018/" + stockName + ".csv");
            System.out.println("Series: " + series.getName() + " (" + series.getSeriesPeriodDescription() + ")");
            System.out.println("Number of bars: " + series.getTickCount());
            System.out.println("First bar: \n"
                    + "\tVolume: " + series.getTick(0).getVolume() + "\n"
                    + "\tOpen price: " + series.getTick(0).getOpenPrice() + "\n"
                    + "\tClose price: " + series.getTick(0).getClosePrice());

                /*
                  Creating indicators
                 */
            // Close price
            ClosePriceIndicator closePrice = new ClosePriceIndicator(series);


            // Relative strength index
            RSIIndicator rsi = new RSIIndicator(closePrice, 14);
            // Williams %R
            WilliamsRIndicator williamsR = new WilliamsRIndicator(series, 14);
            // Simple moving average - SHORT
            SMAIndicator smaShort = new SMAIndicator(closePrice, 12);
            // Simple moving average - LONG
            SMAIndicator smaLong = new SMAIndicator(closePrice, 26);
            // Exponential moving average - SHORT
            EMAIndicator emaShort = new EMAIndicator(closePrice, 12);
            // Exponential moving average - LONG
            EMAIndicator emaLong = new EMAIndicator(closePrice, 26);
            // Weighted moving averages - SHORT
            WMAIndicator wmaShort = new WMAIndicator(closePrice, 12);
            // Weighted moving averages - LONG
            WMAIndicator wmaLong = new WMAIndicator(closePrice, 26);
            // Hull moving averages - SHORT
            HMAIndicator hmaShort = new HMAIndicator(closePrice, 9);
            // Hull moving averages - LONG
            HMAIndicator hmaLong = new HMAIndicator(closePrice, 18);
            // Triple exponential moving average
            TripleEMAIndicator tripleEma = new TripleEMAIndicator(closePrice, 26);
            // Commodity channel index
            CCIIndicator cci = new CCIIndicator(series, 20);
            // Chande momentum oscillator
            CMOIndicator cmo = new CMOIndicator(closePrice, 20);
            // Moving average convergence divergence
            MACDIndicator macd = new MACDIndicator(closePrice, 12, 26);
            // Percentage price oscillator
            PPOIndicator ppo = new PPOIndicator(closePrice, 12, 26);
            // Rate of change
            ROCIndicator roc = new ROCIndicator(closePrice, 21);
            // Chaikin money flow
            ChaikinMoneyFlowIndicator cmfi = new ChaikinMoneyFlowIndicator(series, 21);
            // Directional movement
            DirectionalMovementIndicator dmi = new DirectionalMovementIndicator(series, 14);
            // Parabolic SAR
            ParabolicSarIndicator psi = new ParabolicSarIndicator(series, 14);


            /*
              Building header
             */
            StringBuilder sb = new StringBuilder("timestamp,close,rsi,williamsR,smaShort,smaLong,emaShort,emaLong,wmaShort,wmaLong,hmaShort,hmaLong,tripleEma,cci,cmo,macd,ppo,roc,cmfi,dmi,psi\n");

            /*
              Adding indicators values
             */
            final int nbBars = series.getTickCount();
            for (int j = 0; j < nbBars; j++) {
                Double currentCloseValue = closePrice.getValue(j).toDouble();
                sb.append(series.getTick(j).getEndTime()).append(',')
                        .append(closePrice.getValue(j)).append(',')
                        .append((rsi.getValue(j).toDouble()/50)-1).append(',')
                        .append((williamsR.getValue(j).toDouble()/50)+1).append(',')
                        .append((currentCloseValue-smaShort.getValue(j).toDouble())*10/currentCloseValue).append(',')
                        .append((currentCloseValue-smaLong.getValue(j).toDouble())*10/currentCloseValue).append(',')
                        .append((currentCloseValue-emaShort.getValue(j).toDouble())*10/currentCloseValue).append(',')
                        .append((currentCloseValue-emaLong.getValue(j).toDouble())*10/currentCloseValue).append(',')
                        .append((currentCloseValue-wmaShort.getValue(j).toDouble())*10/currentCloseValue).append(',')
                        .append((currentCloseValue-wmaLong.getValue(j).toDouble())*10/currentCloseValue).append(',')
                        .append((currentCloseValue-hmaShort.getValue(j).toDouble())*10/currentCloseValue).append(',')
                        .append((currentCloseValue-hmaLong.getValue(j).toDouble())*10/currentCloseValue).append(',')
                        .append((currentCloseValue-tripleEma.getValue(j).toDouble())*10/currentCloseValue).append(',')
                        .append(cci.getValue(j).toDouble()/100).append(',')
                        .append(cmo.getValue(j).toDouble()/100).append(',')
                        .append(macd.getValue(j).toDouble()).append(',')
                        .append(ppo.getValue(j).toDouble()).append(',')
                        .append(roc.getValue(j).toDouble()).append(',')
                        .append(cmfi.getValue(j)).append(',')
                        .append((dmi.getValue(j).toDouble()/50)-1).append(',')
                        .append(psi.getValue(j).toDouble()).append('\n');
            }

            /*
              Writing CSV file
             */
            BufferedWriter writer = null;
            try {
                writer = new BufferedWriter(new FileWriter("indicators/"+stockName+"_indicators.csv"));
                writer.write(sb.toString());
            } catch (IOException ioe) {
                Logger.getLogger(CnnTaIndicators.class.getName()).log(Level.SEVERE, "Unable to write CSV file", ioe);
            } finally {
                try {
                    if (writer != null) {
                        writer.close();
                    }
                } catch (IOException ioe) {
                    ioe.printStackTrace();
                }
            }
            System.out.println(stockName+" DONE.");
        }
    }
}