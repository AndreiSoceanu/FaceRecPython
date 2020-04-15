package gui;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author andrei
 */
public class MainWindow extends javax.swing.JFrame {
    String root = "../";
    String personsPath = root + "ConfigData/persons.data";
    String addNewPath = root + "CorePackage/newPerson.py";
    String trainBasePath = root + "TrainBase/";
    String encodingsPath = root + "Encodings/";
    String openCrtPath = root + "CorePackage/openCrt.py";
    
    Set<String> nameSet = new TreeSet<>();
    File file = new File(personsPath);

    /**
     * Creates new form MainWindow
     */

    public MainWindow() {
        initComponents();
        updateSet();

    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        openInCurrentConfigurationButton = new javax.swing.JButton();
        addNewPersonButton = new javax.swing.JButton();
        removePersonButton = new javax.swing.JButton();
        listDatabaseButton = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        chat = new javax.swing.JTextArea();
        jLabel1 = new javax.swing.JLabel();
        clrChat = new javax.swing.JButton();
        reset = new javax.swing.JButton();
        addTextField = new javax.swing.JTextField();
        removeTextField = new javax.swing.JTextField();
        claheWidth = new javax.swing.JTextField();
        jLabel2 = new javax.swing.JLabel();
        filterWidth = new javax.swing.JTextField();
        jLabel3 = new javax.swing.JLabel();
        filterBlur = new javax.swing.JTextField();
        jLabel4 = new javax.swing.JLabel();
        filterColor = new javax.swing.JTextField();
        jLabel5 = new javax.swing.JLabel();
        jLabel6 = new javax.swing.JLabel();
        jLabel7 = new javax.swing.JLabel();
        jLabel8 = new javax.swing.JLabel();
        jLabel9 = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        openInCurrentConfigurationButton.setText("OPEN IN CURRENT CONFIGURATION");
        openInCurrentConfigurationButton.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                openInCurrentConfigurationButtonMouseClicked(evt);
            }
        });

        addNewPersonButton.setText("ADD NEW PERSON TO DATABASE");
        addNewPersonButton.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                addNewPersonButtonMouseClicked(evt);
            }
        });
        addNewPersonButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                addNewPersonButtonActionPerformed(evt);
            }
        });

        removePersonButton.setText("REMOVE PERSON(S) FROM DATABASE");
        removePersonButton.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                removePersonButtonMouseClicked(evt);
            }
        });

        listDatabaseButton.setText("LIST DATABASE");
        listDatabaseButton.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                listDatabaseButtonMouseClicked(evt);
            }
        });

        chat.setEditable(false);
        chat.setColumns(20);
        chat.setRows(5);
        jScrollPane1.setViewportView(chat);

        jLabel1.setText("MESSAGES FROM THE APPLICATION");

        clrChat.setText("CLEAR MESSAGEBOX");
        clrChat.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                clrChatMouseClicked(evt);
            }
        });

        reset.setText("RESET APPLICATION");

        claheWidth.setText("4");

        jLabel2.setText("CLAHE window width");

        filterWidth.setText("3");

        jLabel3.setText("Filter window width");

        filterBlur.setText("75");

        jLabel4.setText("Filter blur strength");

        filterColor.setText("75");

        jLabel5.setText("Filter color sigma");

        jLabel6.setText("Defalut value: 4");

        jLabel7.setText("Default value: 3");

        jLabel8.setText("Default value: 75");

        jLabel9.setText("Default value: 75");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jScrollPane1)
                        .addContainerGap())
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(addNewPersonButton, javax.swing.GroupLayout.DEFAULT_SIZE, 257, Short.MAX_VALUE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(addTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 241, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(javax.swing.GroupLayout.Alignment.LEADING, layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                        .addComponent(removePersonButton)
                                        .addComponent(jLabel2)
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                            .addComponent(jLabel4)
                                            .addComponent(jLabel3)))
                                    .addComponent(jLabel5, javax.swing.GroupLayout.Alignment.TRAILING))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(removeTextField)
                                    .addGroup(layout.createSequentialGroup()
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                            .addComponent(filterColor, javax.swing.GroupLayout.DEFAULT_SIZE, 52, Short.MAX_VALUE)
                                            .addComponent(filterBlur)
                                            .addComponent(filterWidth)
                                            .addComponent(claheWidth))
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                            .addComponent(jLabel6)
                                            .addComponent(jLabel7)
                                            .addComponent(jLabel8)
                                            .addComponent(jLabel9))
                                        .addGap(0, 0, Short.MAX_VALUE)))))
                        .addGap(30, 30, 30))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(listDatabaseButton, javax.swing.GroupLayout.PREFERRED_SIZE, 257, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(0, 0, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(clrChat)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(reset)
                        .addGap(91, 91, 91))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(openInCurrentConfigurationButton, javax.swing.GroupLayout.PREFERRED_SIZE, 257, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(layout.createSequentialGroup()
                                .addGap(156, 156, 156)
                                .addComponent(jLabel1)))
                        .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(27, 27, 27)
                .addComponent(openInCurrentConfigurationButton)
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(addNewPersonButton)
                    .addComponent(addTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(removePersonButton)
                    .addComponent(removeTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(claheWidth, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel2)
                    .addComponent(jLabel6))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(filterWidth, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel3)
                    .addComponent(jLabel7))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(filterBlur, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel4)
                    .addComponent(jLabel8))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(filterColor, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel5)
                    .addComponent(jLabel9))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 13, Short.MAX_VALUE)
                .addComponent(listDatabaseButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jLabel1)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 142, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(clrChat)
                    .addComponent(reset))
                .addGap(16, 16, 16))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void addNewPersonButtonMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_addNewPersonButtonMouseClicked
        
        String name = addTextField.getText().replace(" ", "");
        if(name.equals("")) return;
        if (nameSet.contains(name)) {
            String msg = chat.getText();
            msg += "Label " + name + " already exists! Routine interrupted!\n";
            chat.setText(msg);
            return;
        }
        
        nameSet.add(name);
        int size = nameSet.size();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            StringBuilder str = new StringBuilder();
            str.append(Integer.toString(size));
            str.append('\n');
            nameSet.stream().map((x) -> {
                str.append(x);
                return x;
            }).forEachOrdered((_item) -> {
                str.append('\n');
            });
            fos.write(str.toString().getBytes());
            addTextField.setText("");
            int cw, fw, fb, fs;
            try{
                cw = Integer.parseInt(claheWidth.getText());
                fw = Integer.parseInt(filterWidth.getText());
                fb = Integer.parseInt(filterBlur.getText());
                fs = Integer.parseInt(filterColor.getText());
            }
            catch(NumberFormatException e){
                chat.setText(chat.getText() + "Invalid numerical parametters!\n");
                return;
            }
            String cmd = "python " + addNewPath + " " + name + " " + cw + " " + fw + " " + fb + " " + fs;
            
            Process p = Runtime.getRuntime().exec(cmd);
            BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String ret = in.readLine();
            if (ret.equalsIgnoreCase("Done")) {
                this.chat.setText(this.chat.getText() + "Added " + name + " in the database.\n");
            }
            else {
                nameSet.remove(name);
            }
        } catch (IOException ex) {
            Logger.getLogger(MainWindow.class.getName()).log(Level.SEVERE, null, ex);
        }
    }//GEN-LAST:event_addNewPersonButtonMouseClicked

    private void clrChatMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_clrChatMouseClicked
        chat.setText("");
    }//GEN-LAST:event_clrChatMouseClicked

    private void addNewPersonButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_addNewPersonButtonActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_addNewPersonButtonActionPerformed

    private void removePersonButtonMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_removePersonButtonMouseClicked
        String name = this.removeTextField.getText().replaceAll(" ", "");
        if(name.equals("")) return;
        if (!nameSet.contains(name)) {
            chat.setText(chat.getText() + "Label "+name + " does not exist. Nothing removed.\n");
            this.removeTextField.setText("");
            return;
        }
        
        nameSet.remove(name);
        int size = nameSet.size();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            StringBuilder str = new StringBuilder();
            str.append(Integer.toString(size));
            str.append('\n');
            nameSet.stream().map((x) -> {
                str.append(x);
                return x;
            }).forEachOrdered((_item) -> {
                str.append('\n');
            });
            fos.write(str.toString().getBytes());
        } catch (IOException ex) {
            Logger.getLogger(MainWindow.class.getName()).log(Level.SEVERE, null, ex);
        }
        removeTextField.setText("");
        String cmd1 = "rm -rf " + trainBasePath + name + ".jpg";
        String cmd2 = "rm -rf " + encodingsPath + name + ".data";
        
        try {
            Process p1 = Runtime.getRuntime().exec(cmd1);
            Process p2 = Runtime.getRuntime().exec(cmd2);
        } catch (IOException ex) {
            Logger.getLogger(MainWindow.class.getName()).log(Level.SEVERE, null, ex);
        }
        chat.setText(chat.getText()+"Removed " + name + " from database.\n");
    }//GEN-LAST:event_removePersonButtonMouseClicked

    private void listDatabaseButtonMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_listDatabaseButtonMouseClicked
        String list = "=======\nDatabase:\n";
        list = nameSet.stream().map((name) -> name + "\n").reduce(list, String::concat);
        chat.setText(chat.getText()+list);
    }//GEN-LAST:event_listDatabaseButtonMouseClicked

    private void openInCurrentConfigurationButtonMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_openInCurrentConfigurationButtonMouseClicked
        int cw, fw, fb, fs;
        try{
            cw = Integer.parseInt(claheWidth.getText());
            fw = Integer.parseInt(filterWidth.getText());
            fb = Integer.parseInt(filterBlur.getText());
            fs = Integer.parseInt(filterColor.getText());
        }
        catch(NumberFormatException e){
            chat.setText(chat.getText() + "Invalid numerical parametters!\n");
            return;
        }
        
        String cmd1 = "python " + openCrtPath + " " + cw + " " + fw + " " + fb + " " + fs;
        try {
            Process p1 = Runtime.getRuntime().exec(cmd1);
        } catch (IOException ex) {
            Logger.getLogger(MainWindow.class.getName()).log(Level.SEVERE, null, ex);
        }
    }//GEN-LAST:event_openInCurrentConfigurationButtonMouseClicked

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(() -> {
            new MainWindow().setVisible(true);
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton addNewPersonButton;
    private javax.swing.JTextField addTextField;
    private javax.swing.JTextArea chat;
    private javax.swing.JTextField claheWidth;
    private javax.swing.JButton clrChat;
    private javax.swing.JTextField filterBlur;
    private javax.swing.JTextField filterColor;
    private javax.swing.JTextField filterWidth;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JLabel jLabel6;
    private javax.swing.JLabel jLabel7;
    private javax.swing.JLabel jLabel8;
    private javax.swing.JLabel jLabel9;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JButton listDatabaseButton;
    private javax.swing.JButton openInCurrentConfigurationButton;
    private javax.swing.JButton removePersonButton;
    private javax.swing.JTextField removeTextField;
    private javax.swing.JButton reset;
    // End of variables declaration//GEN-END:variables

    private void updateSet() {
        try {
            Scanner scan = new Scanner(file);
            int n = Integer.parseInt(scan.nextLine());
            for (int i = 0; i < n; i++) {
                nameSet.add(scan.nextLine());
            }
        } catch (FileNotFoundException ex) {
            String msg = chat.getText();
            msg += "\nCould not read persons.data!\n";
            chat.setText(msg);
            Logger.getLogger(MainWindow.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
