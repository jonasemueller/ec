all:
	rm -f data/geom/logoDrawString
	cd solvers && \
	  jbuilder build solver.exe && \
	  jbuilder build versionDemo.exe && \
	  jbuilder build helmholtz.exe && \
	  jbuilder build logoDrawString.exe && \
	  jbuilder build protonet-tester.exe && \
	  jbuilder build compression.exe && \
	  jbuilder build evolution.exe && \
	  cp _build/default/compression.exe ../compression && \
	  cp _build/default/versionDemo.exe ../versionDemo && \
	  cp _build/default/evolution.exe ../evolution && \
	  cp _build/default/solver.exe ../solver && \
	  cp _build/default/helmholtz.exe ../helmholtz && \
	  cp _build/default/protonet-tester.exe ../protonet-tester && \
	  cp _build/default/logoDrawString.exe \
	    ../logoDrawString && \
	  ln -s ../../logoDrawString \
	    ../data/geom/logoDrawString
			
copy:
				cp ../ec_language/ec_language/solver .
				cp ../ec_language/ec_language/compression .
				cp ../ec_language/ec_language/helmholtz .
				cp ../ec_language/ec_language/logoDrawString .
				cp ../ec_language/ec_language/data/geom/logoDrawString data/geom/logoDrawString
clean:
	cd solvers && jbuilder clean
	rm -f solver
	rm -f compression
	rm -f helmholtz
	rm -f logoDrawString
	rm -f data/geom/logoDrawString

clevrTestClean: 
	cd solvers && jbuilder clean
	rm -f clevrTest

clevrTest:
	cd solvers && \
	jbuilder build clevrTest.exe && \
	cp _build/default/clevrTest.exe ../clevrTest
	
re2TestClean: 
	cd solvers && jbuilder clean
	rm -f re2Primitives
	
re2Test:
		cd solvers && \
		jbuilder build re2Primitives.exe && \
		cp _build/default/re2Primitives.exe ../re2Primitives