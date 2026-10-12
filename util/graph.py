from feature.util.data_utils import pb_str_to_datetime
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from tqdm import tqdm

zero_dt = timedelta(seconds=0)

class DataGraph:
    """
    Abstraction for a data graph. A data graph contains identified features
    as nodes, and these are connected by edges, when they represent the same
    entity as resulted by a pairwise comparison of the features.
    
    It contains methods to iterate a graph and to generate the tracks.
    """

    def __init__(self, connections, tr_tech=None, fix=False, remove_transitive=False, is_track=False):
        """
        Creates a new graph from the given connections as list of edges. The reference to the tracking technique 
        is needed to call adjust_track when computing tracks.You can also pass None if you don't plan on generating tracks.
        
        Parameters
        ----------
        connections : list of protobuf_reference.GraphConnection()
            The edges as list of GraphConnections. You can use the `tr_tech.get_new_connection()` to generate these edges.
        tr_tech : TrackingStrategy, optional
            Reference to the tracking strategy. Needed when generating tracks to call the tracking's `adjust_track()`.
        fix : bool, optional
            Whether the graph should be fixed upon creation, e.g. delete dangling edges.
        remove_transitive : bool, optional
            Whether transitive edges should be removed on creation.
        is_track : bool, optional
            Whether the given `connections` already form a track.
        """
        
        # self.set_desc = set_desc
        self.edges = connections
        self.edges.sort(key=lambda item: item.parent.time)

        self.tr_tech = tr_tech
        
        if fix:
            self.fix_graph()
            self.edges.sort(key=lambda item: item.parent.time)
            
        # dict for quick access of edges based on feature id
        self.edge_dict = dict()
        for edge in self.edges:
            self.edge_dict[edge.parent.feature.id] = edge
            
        if remove_transitive:
            self.remove_transitive()
            for edge in self.edges: # recompute
                self.edge_dict[edge.parent.feature.id] = edge
        
        # get list of nodes from edge list
        node_ids = []
        nodes = []
        
        for edge in self.edges:
            if edge.parent.feature.id not in node_ids:
                node_ids.append(edge.parent.feature.id)
                nodes.append(edge.parent)
            else:
                tqdm.write("Given edge set as multiple similar node IDs. Skipping node.")
            
        self.nodes = nodes
        
        
        if is_track:
            self.tracks = [self] # already set as track
        else:
            self.tracks = None
        
    @staticmethod
    def from_pb2(obj_set, tr_tech=None):
        """Creates a data graph from a given protobuf object.

        Parameters
        ----------
        obj_set: pb2.TrackableSet()
            The protobuf set.
        tr_tech: TrackingTechnique
            The tracking algorithm reference. If not provided, it won't be possible to generate tracks.

        Returns
        -------
        DataGraph
            The data graph.

        """
        edges_pb2 = obj_set.graph.edges
        tracks_pb2 = obj_set.tracks
        
        dg = DataGraph(edges_pb2, tr_tech=tr_tech)
        
        # already fill dg.tracks if they have been computed and are present
        track_graphs = []
        for track in tracks_pb2:
            
            t = DataGraph(track.edges, tr_tech, is_track=True)
            track_graphs.append(t)
        
        dg.tracks = track_graphs
        return dg
        
    
    def get_earliest_nodes(self):
        """
        Get the nodes which are earliest in the graph.
        Note that multiple earliest nodes might exist.

        Returns
        -------
        List of nodes with earliest timestamp.
        """
        if len(self.edges) == 0:
            return []

        earliest_time = self.edges[0].parent.time
        nodes = [self.edges[0].parent]

        # iterate over first nodes until time changes or end of list
        idx = 1
        while idx < len(self.edges) and self.edges[idx].parent.time == earliest_time:
            nodes.append(self.edges[idx].parent)
            idx += 1

        return nodes

    def get_latest_nodes(self):
        """
        Get the nodes which are latest in the graph.
        Note that multiple latest nodes might exist.

        Returns
        -------
        List of nodes with latest timestamp.
        """
        if len(self.edges) == 0:
            return []

        latest_time = self.edges[-1].parent.time
        nodes = [self.edges[-1].parent]

        # iterate over first nodes until time changes or end of list
        idx = len(self.edges) - 2
        while idx >= 0 and self.edges[idx].parent.time == latest_time:
            nodes.append(self.edges[idx].parent)
            idx -= 1

        return nodes

    def fix_graph(self):
        """
        Fixes graph by removing orphaned edges. (Edges where endnode is not a valid node)

        Returns
        -------
        """
        nodes = [e.parent for e in self.edges]

        for edge in self.edges:
            for child in edge.children[:]:
                if child not in nodes:
                    edge.children.remove(child)

    def remove_transitive(self):
        """ Removes transitive edges in this graph.
        
        Returns
        -------
        """
        
        tn = 0
        # an edge is transitive (a->c if a->b->c exists), when the downstream nodes of "a" without this edge contains the end node ("c")
        for con in self.edges:

            children = con.children[:]
            # cant be transitive edge if only one child
            if len(children) < 2:
                continue
            for cld in children:
                child_time = pb_str_to_datetime(cld.time)
                con.children.remove(cld)  # remove edge temporarily from DataGraph
                # and check if downstream nodes still contains end node of edge. if yes -> transitive
                downstream_ids = self.get_downstream_node_ids(con.parent, until_time=child_time)
                
                # downstream_objs = [self.edge_dict[ds_id].parent for ds_id in downstream_ids]
                if cld.feature.id in downstream_ids:
                    # transitive
                    tn = tn + 1
                else:
                    # add edge back
                    con.children.append(cld)
        
        # Nodes and edgedict does not change, so we dont need to do further updates
        if tn > 0:
            tqdm.write("Removed " + str(tn) + " transitive edges in current set.")
    
    def get_downstream_node_ids(self, node, until_time=None):
        """
        Helper method for the track generation. Searches all downstream node indices.
        Until the until_time.

        Parameters
        ----------
        node: pb2.GraphNode()
            Node in self.nodes to start from.
        until_time: datetime
            Until when the nodes should be returned
            
        Returns
        -------
        list of downstream indices in list
        """
        
        node_ids = [node.feature.id]
        edge = self.edge_dict[node.feature.id]

        node, connected_nodes = edge.parent, edge.children

        for c_node in connected_nodes:
            # if until_time is set: if c_node time later than until_time end recursion here
            if until_time is not None and pb_str_to_datetime(c_node.time) > until_time:
                continue

            # for edge in self.edges:
            #     self.edge_dict[edge.parent.feature.id] = edge
            # get index of connected node in graph
            try:
                c_node_id = c_node.feature.id
                c_parent_node = self.edge_dict[c_node_id].parent
            except ValueError:
                print("Could not find children node as a parent node. Skip.")
                continue

            # call recursively on this connected node
            c_node_downstream_ids = self.get_downstream_node_ids(c_parent_node, until_time=until_time)
            node_ids.extend(c_node_downstream_ids)

        return list(set(node_ids))
        
        
    def get_nodes_after(self, time_delta, start_index):
        """Get all nodes `time_delta` after the node at `start_index`, as dict, with the
        keys being the end nodes and values all nodes along the way.
        
        Parameters
        ----------
        start_index: int
            Start index referencing starting node in self.nodes
        time_delta: datetime
            How much later to start_index node the end nodes should be returned.
            
        Returns
        -------
        dict with keys being the end nodes, so after `time_delta`, and values are all
        nodes along the way.
        
        """
        # print(time_delta)

        this_node = self.edges[start_index].parent
        if time_delta <= zero_dt or len(self.edges[start_index].children) == 0:
            dd = defaultdict(list)
            dd[start_index] = [start_index]
            return dd

        obj_node_list = [e.parent for e in self.edges]
        children_ = defaultdict(list)

        for child in self.edges[start_index].children:
            # print(child.time)
            dt = pb_str_to_datetime(child.time) - pb_str_to_datetime(this_node.time)
            remaining_time_delta = time_delta - dt

            # get index of connected node in graph
            c_node_idx = obj_node_list.index(child)
            after_nodes = self.get_nodes_after(remaining_time_delta, c_node_idx)
            # add own node to it (this_node)
            for endn_, pathn_ in after_nodes.items():
                children_[endn_].extend(pathn_ + [start_index])

        return children_

    
    
    def get_parents(self, node):
        """
        Get the parents of a node (previous timestep)
        Parameters
        ----------
        node: pb2.GraphNode
            The node to get the parents of

        Returns
        -------
        List of nodes representing the parents. Multiple nodes indicate a merge event.
        """
        parents = []
        for edge in self.edges:
            for child in edge.children:
                if child == node:
                    parents.append(edge.parent)
        return parents

    def get_children(self, node):
        """
        Get the children of a node (next timestep)
        
        Parameters
        ----------
        node: pb2.GraphNode
            The node to get the children of

        Returns
        -------
        List of nodes representing the children. Multiple nodes indicate a split event.
        """
        for edge in self.edges:
            if edge.parent == node:
                return edge.children
        return []

    
    def fill_protobuf(self, obj_set):
        """ Fill this graph into the given protobuf object. 
        
        Parameters
        ----------
        obj_set: pb2.TrackableSet
            The protobuf set to include in this graph.

        Returns
        -------
        """
        
        # add graph nodes to ref graph
        obj_set.graph.edges.extend(self.edges)
        
        # build the graph and tracks into the protobuf object
        for track_id, track_ in enumerate(self.tracks):
            track_obj = self.tr_tech.pb_reference.FeatureGraph()
            track_obj.edges.extend(track_.edges)
            obj_set.tracks.append(track_obj)
            
   # ------------------------------------------------------------------ DEBUG
    def _dbg_graph_summary(self):
        """Print node/edge counts and time coverage of this graph (before track generation)."""
        n_nodes = len(self.edges)
        n_with_children = sum(1 for e in self.edges if len(e.children) > 0)
        n_links = sum(len(e.children) for e in self.edges)
 
        times = sorted({pb_str_to_datetime(e.parent.time) for e in self.edges})
        cfg_ = getattr(self.tr_tech, 'config', None)
        max_cmp = getattr(cfg_, 'max_cmp_delta', timedelta(hours=12))
 
        print("\n[DEBUG graph] ---------------------------------------------")
        print(f"  nodes (detections):         {n_nodes}")
        print(f"  nodes with >=1 child:       {n_with_children}")
        print(f"  nodes with no child:        {n_nodes - n_with_children}")
        print(f"  total edges (child links):  {n_links}")
        print(f"  distinct timesteps w/ nodes:{len(times)}")
        if len(times) > 1:
            gaps = [b - a for a, b in zip(times, times[1:])]
            n_big = sum(1 for g in gaps if g > max_cmp)
            print(f"  first/last time:            {times[0]} .. {times[-1]}")
            print(f"  min/max gap between timesteps with detections: {min(gaps)} / {max(gaps)}")
            print(f"  gaps > max_cmp_delta ({max_cmp}): {n_big} of {len(gaps)}  <- each breaks tracks")
        print("-----------------------------------------------------------\n")
 
    @staticmethod
    def _dbg_len_hist(tracks, label):
        lens = [len(t.edges) for t in tracks]
        buckets = Counter()
        for l in lens:
            key = '1' if l == 1 else '2' if l == 2 else '3-4' if l <= 4 else '5-8' if l <= 8 else '9+'
            buckets[key] += 1
        order = ['1', '2', '3-4', '5-8', '9+']
        print(f"[DEBUG tracks] {label}: {len(tracks)} tracks; length [nodes] histogram: "
              + ", ".join(f"{k}: {buckets[k]}" for k in order))
    # ---------------------------------------------------------------- /DEBUG
            
            
    def generate_tracks(self, apply_filter=True, depth=0):
        """
        Generate tracks from edge list in this DataGraph.
        This track generation is based on the longest path invariant:
        A track is always continued along the longest possible time until dissipation
        of the feature. 

        Parameters
        ----------
        apply_filter: bool
            Whether the tracks should be filtered by the tracking algorithm. If True,
            `adjust_track()` is called for each track.
            
        Returns
        -------
        Nothing, but self.tracks will be filled.
        """
        
        # note ID != IDX -> ID is node id given by user, idx is for list indices
        # ids can also be negative or so
        if depth == 0:
            tqdm.write(f"Start generate tracks...")
        if self.tr_tech is None:
            tqdm.write(f"Tracking Technique not provided to graph, cant create tracks. Abort.")
            return
            
        if depth == 0:  # DEBUG
            self._dbg_graph_summary()
            if hasattr(self.tr_tech, 'dbg_report_correspond'):
                self.tr_tech.dbg_report_correspond()
        
        time_sorted_edges = list(sorted(self.edges, key=lambda item: item.parent.time))
        
        node_discovered = dict() # [False] * len(self.nodes)
        time_until_dissipation = dict()
        track_id_per_node = dict()
        for edge in time_sorted_edges:
            node_discovered[edge.parent.feature.id] = False
            time_until_dissipation[edge.parent.feature.id] = None
            track_id_per_node[edge.parent.feature.id] = None
        
        def child_dt(edge, child):
            return pb_str_to_datetime(child.time) - pb_str_to_datetime(edge.parent.time)
        
        def DFS1(edge_id):
            # time_until_dissipation[n] = duration of the longest path starting at n (memoized).
            # A child that was already visited from another parent is reused, not skipped:
            # skipping it underestimated this node's duration and cut its path short.
            node_discovered[edge_id] = True
            edge = self.edge_dict[edge_id]
            
            longest = timedelta(seconds=0)
            for child in edge.children:
                child_id = child.feature.id
                if not node_discovered[child_id]:
                    DFS1(child_id)
                if time_until_dissipation[child_id] is None:
                    continue  # still being visited; cannot happen as edges point forward in time
                # include the step to the child, so a +6h and a +12h child compare fairly
                longest = max(longest, child_dt(edge, child) + time_until_dissipation[child_id])
            
            time_until_dissipation[edge_id] = longest
            
        def indexing(edge_id, track_idx):
            # follow the longest continuation among children not yet claimed by an earlier track
            while edge_id is not None:
                track_id_per_node[edge_id] = track_idx
                edge = self.edge_dict[edge_id]
                free = [c for c in edge.children if track_id_per_node[c.feature.id] is None]
                if not free:
                    break
                best = max(free, key=lambda c: child_dt(edge, c) + time_until_dissipation[c.feature.id])
                edge_id = best.feature.id
            
        
        for edge in time_sorted_edges:
            edge_id = edge.parent.feature.id
            if not node_discovered[edge_id]:
                DFS1(edge_id)
                
        
        track_idx = 0
        for edge in time_sorted_edges:
            edge_id = edge.parent.feature.id
            if track_id_per_node[edge_id] is None:
                # new track
                indexing(edge_id, track_idx)
                track_idx += 1
                
        
        self.build_filter_tracks(time_sorted_edges, track_id_per_node, apply_filter, track_idx, depth=depth)
        
        if depth == 0:
            tqdm.write("Done generating tracks.")
            
        if apply_filter:
                from feature.identification.african_easterly_waves.hovmoller_plots import plot_hovmoller_tracks
                cfg_ = getattr(self.tr_tech, 'config', None)
                plot_hovmoller_tracks(
                    self.tracks,
                    ds=getattr(self.tr_tech, 'hov_ds', None),
                    rejected_tracks=getattr(self, 'dbg_prefilter_tracks', None),
                    cfg=cfg_,
                )
            
        return
        
        
    def build_filter_tracks(self, time_sorted_edges, track_id_per_node, apply_filter, num_tracks, depth=0):
        # Helper func for track gen: given edges, ids per node, apply filters and build tracks into self.
        
        # put them into track bins
        tracks = [[] for i in range(num_tracks)]
        
        for edge in time_sorted_edges:
            edge_id = edge.parent.feature.id
            track_id = track_id_per_node[edge_id]
            
            tracks[track_id].append(edge)
            
        track_graphs = []
        for track in tracks:
            # to Graph. fixes dangling edges
            dg = DataGraph(track, self.tr_tech, fix=True, is_track=True)
            track_graphs.append(dg)
            
        
        # first set of tracks
        self.tracks = track_graphs
        
        if apply_filter:  # DEBUG
            self._dbg_len_hist(self.tracks, "before adjust_track")
        
        edges_before_filter = []
        for track in self.tracks:
            edges_before_filter.extend(track.edges)
        
        if not apply_filter:
            return
        
        i = 0
        while i < 3: # do while
            edges_after_filter = []

            for track in self.tracks:
                # let user change track, reject it, or split it
                adj_track = self.tr_tech.adjust_track(track)

                if adj_track is None:
                    pass # user rejected it
                elif type(adj_track) == DataGraph:
                    # accept as it is
                    edges_after_filter.extend(track.edges)

                elif type(adj_track) == bool and adj_track: # accept via bool
                    edges_after_filter.extend(track.edges)

                elif type(adj_track) == list:
                    # list of edges that got accepted: might have split into two tracks! might require recompute, see below
                    edges_after_filter.extend(adj_track)

                else:
                    tqdm.write("Unxpected return of adjust_track(). Expect bool or edge list.")
                    exit(1)
                    
            # DEBUG: report what adjust_track() did in this iteration
            tqdm.write(f"[DEBUG filter iteration {i}] edges before: {len(edges_before_filter)}, "
                       f"edges accepted: {len(edges_after_filter)}")
            if hasattr(self.tr_tech, 'dbg_report_adjust'):
                self.tr_tech.dbg_report_adjust()
            edges_before_filter = list(edges_after_filter)

            # generate tracks with filtered edges.
            dg = DataGraph(edges_after_filter, self.tr_tech, fix=True)
            dg.generate_tracks(apply_filter=False, depth=depth+1)

            # did amount of tracks change?
            tracks_before = len(self.tracks)
            tracks_after = len(dg.tracks)

            self.tracks = dg.tracks
            if apply_filter:
                self.dbg_prefilter_tracks = list(track_graphs)
                self._dbg_len_hist(self.tracks, "before adjust_track")

            if tracks_after == tracks_before:
                break
            else:
                i += 1
                tqdm.write(f"Continue loop iteration since amount of tracks changed: {tracks_before} -> {tracks_after}")
                
        
        
        
    def generate_tracks_uf(self, apply_filter=True, depth=0):
        """
        Generate tracks from edge list in this DataGraph.
        This track generation uses the union-find strategy to find 
        subgraphs. Therefore one track is defined as one subgraph in the graph.

        Parameters
        ----------
        apply_filter: bool
            Whether the tracks should be filtered by the tracking algorithm. If True,
            `adjust_track()` is called for each track.
            
        Returns
        -------
        Nothing, but self.tracks will be filled.
        """
        
        """
        Union find version of track extraction: Track = connected subgraph
        """
        # note ID != IDX -> ID is node id given by user, idx is for list indices
        # ids can also be negative or so
        print("Start generate tracks")
        
        time_sorted_edges = list(sorted(self.edges, key=lambda item: item.parent.time))
        
        node_discovered = dict() # [False] * len(self.nodes)
        for edge in time_sorted_edges:
            node_discovered[edge.parent.feature.id] = False
        
        track_id_per_node = dict()
        parent = dict()
        sizes = dict()
        for edge in time_sorted_edges:
            id_ = edge.parent.feature.id
            track_id_per_node[id_] = None
            parent[id_] = id_
            sizes[id_] = 1
            
        global subgraph_count
        subgraph_count = len(time_sorted_edges) # start with |E| subgraphs
        
        print("Subgraphs before UF " + str(subgraph_count))
        
        def find(node):
            global subgraph_count
            while node != parent[node]:
                # path compression
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node
        
        def union(node1, node2):
            global subgraph_count
            
            root1 = find(node1)
            root2 = find(node2)
            
            # already in the same set
            if root1 == root2:
                return
            
            if sizes[root1] > sizes[root2]:
                parent[root2] = root1
                sizes[root1] += 1
            else:
                parent[root1] = root2
                sizes[root2] += 1
            
            subgraph_count -= 1
        
        for edge in time_sorted_edges:
            for ch in edge.children:
                union(edge.parent.feature.id, ch.feature.id)
            
        
        print("Subgraphs after UF " + str(subgraph_count))
        
        track_idx = 0
        to_int_keys = dict()
        for edge in time_sorted_edges:
            edge_id = edge.parent.feature.id
            
            p = parent[edge_id]
            if p not in to_int_keys.keys():
                to_int_keys[p] = track_idx
                track_idx += 1
            track_id_per_node[edge_id] = to_int_keys[p]
            
        
        self.build_filter_tracks(time_sorted_edges, track_id_per_node, apply_filter, track_idx, depth=depth)
        
        tqdm.write("Done generating tracks.")
        
        return
    