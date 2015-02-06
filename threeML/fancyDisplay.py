class HtmlTable(object):
  def __init__(self,ncols):
    self.ncols                = ncols
    self.rows                 = []
    self.headings             = None
  pass
  
  def addRow(self,*args):
    if(len(args)!=self.ncols):
      raise RuntimeError("Error in constructing row for table: wrong number of elements")
    else:
      self.rows.append(list(args))
    pass
  pass
  
  def addHeadings(self,*args):
    self.headings             = list(args)
  pass
  
  def __repr__(self):
    output                    = []
    output.append("<table>")
    if(self.headings!=None):
      output.append("<tr><th>"+"</th><th>".join(map(str,self.headings))+"</th></tr>")
    pass
    
    for row in self.rows:
      output.append("<tr><td>"+"</td><td>".join(map(str,row))+"</td></tr>")
    pass
    
    output.append(r"</table>")
    return "\n".join(output)
  pass
pass
