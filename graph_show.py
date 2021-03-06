import random
class GraphShow():
    """"Create demo page"""
    def __init__(self):
        self.base = '''
    
    <html>
    <head>
      <title>Knowledge Graph Generation</title>
      <script type="text/javascript" src="../static/VIS/dist/vis.js"></script>
      <link href="../static/VIS/dist/vis.css" rel="stylesheet" type="text/css">
      <link href="../static/css/kg.css" rel="stylesheet" type="text/css">
      <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    </head>
    <body>
    <div id="top_head">
    <span id="head_text">
        Knowledge Graph Generation Demo
    </span>
    </div>
    <div id="box_container">
        <div id="div_form">
            <form id="kg_info" action="." method="post" enctype=multipart/form-data>
                
                <label for="textInput" id="message">--Input Text--</label>
                <textarea id="textInput" name="textInput" rows="30" cols="60">input text here.
                </textarea>
                <br><br>

                <div id="pt1">
                <label for="myfile">Import Relation Schema (Optional)</label>
                <input type="file" id="myfile" name="myfile">
                </div>
                
                
                <div id="pt2">
                <span>Select Optional Entity Type:</span>
                <input type="checkbox" id="enttype2" name="enttype2" value="NOUN">
                <label for="enttype2"> Noun Phrases </label>
                </div>
                
                
                <input type="submit" value="Submit">
                
            </form>
        </div>
        <div id="VIS_draw"></div>
    </div>

    <script type="text/javascript">
      var nodes = data_nodes;
      var edges = data_edges;

      var container = document.getElementById("VIS_draw");

      var data = {
        nodes: nodes,
        edges: edges
      };

      var options = {

          nodes: {
              
              size: 10,
              font: {
                  size: 25
              }
          },
          edges: {
              font: {
                  size: 25,
                  align: 'center'
              },
              color: 'orange',
              arrows: {
                  to: {enabled: true, scaleFactor: 1.2}
              },
              scaling: {
                  
                  label: {enabled: true}
              },
              length: 300,
              smooth: {enabled: false},
              
          },
          physics: {
              enabled: false,
              solver: "repulsion",
              repulsion: {
                  nodeDistance: 300
              }
          }
      };

      var network = new vis.Network(container, data, options);
      network.stabilize();

    </script>
    </body>
    </html>
    '''
    

    def create_page(self, events, text):
        """Read data"""
        nodes = []
        for event in events:
            nodes.append(event[0])
            nodes.append(event[2])
        node_dict = {node: index for index, node in enumerate(nodes)}

        data_nodes = []
        data_edges = []
        for node, id in node_dict.items():
            data = {}
            data["group"] = 'Event'
            data["id"] = id
            data["label"] = node
            data_nodes.append(data)
        
        for edge in events:
            data = {}
            from_node = []
            to_node = []
            
            data['from'] = node_dict.get(edge[0])
            data['label'] = edge[1]
            data['to'] = node_dict.get(edge[2])
            from_node.append(data['from'])
            to_node.append(data['to'])
            #if data['from'] not in from_node and data['to'] not in to_node:
            data['smooth'] = {'type': 'curvedCW', 'roundness': random.random()}
                
            #else:
            #    data['smooth'] = {'type': 'curvedCW'}
            #    data['roundness'] = 0.2 + random.random()
                
            data_edges.append(data)

        self.create_html(data_nodes, data_edges, text)
        return

    def create_html(self, data_nodes, data_edges, text):
        """Generate html file"""
        f = open('./templates/graph_show.html', 'w+')
        html = self.base.replace('data_nodes', str(data_nodes)).replace('data_edges', str(data_edges)).replace('input text here.', str(text))
        f.write(html)
        f.close()