# Apartado 4.5.2: sistema MVKD (Aitken y Lucy, 2004) con el paquete fvclrr, por vocal, en las cuatro condiciones.
# Suspect/fondo = sesion A; delincuente (offender) = sesion B; la condicion cruzada usa formantes limpios en A y telefonicos en B.
# Salida: resultados/mvkd_pares.csv (logLR por pareja, vocal y condicion). La fusion por vocal y las metricas se calculan en mvkd_evaluacion.py.
# Uso: Rscript codigo/r/mvkd_formantes.R
.libPaths(c("codigo/r", .libPaths()))
suppressMessages(library(fvclrr))
set.seed(42)
d <- read.csv("rasgos/formantes.csv", fileEncoding = "UTF-8")
vocales <- c("a", "e", "i", "o", "u")
MIN_TOKENS <- 5

# hablantes con al menos MIN_TOKENS tokens en cada sesion y vocal (comunes a todas las vocales para poder fusionarlas)
cnt <- table(d$hablante, d$sesion, d$vocal)
hab <- rownames(cnt)[apply(cnt >= MIN_TOKENS, 1, all)]
cat("Hablantes comunes:", length(hab), "\n")
write.csv(data.frame(hablante = hab), "resultados/mvkd_hablantes.csv", row.names = FALSE)

L <- c("F1_limpio", "F2_limpio", "F3_limpio"); T <- c("F1_tel", "F2_tel", "F3_tel")
CONDS <- list(limpio_F1234 = list(A = c(L, "F4_limpio"), B = c(L, "F4_limpio")),
              limpio_F123  = list(A = L, B = L),
              telefono_F123 = list(A = T, B = T),
              cruzado_F123 = list(A = L, B = T))
filas <- list(); t0 <- Sys.time()
for (cn in names(CONDS)) {
  for (v in vocales) {
    A <- d[d$sesion == "A" & d$vocal == v & d$hablante %in% hab, c("hablante", CONDS[[cn]]$A)]
    B <- d[d$sesion == "B" & d$vocal == v & d$hablante %in% hab, c("hablante", CONDS[[cn]]$B)]
    colnames(A)[1] <- "speaker"; colnames(B) <- colnames(A)
    A <- A[complete.cases(A), ]; B <- B[complete.cases(B), ]
    por_hablante <- list(suspect_data = split(A[, -1, drop = FALSE], A$speaker), offender_data = split(B[, -1, drop = FALSE], B$speaker))
    # MVKD de fvclrr (el fondo son los datos de la sesion A de los demas hablantes)
    r <- MVKD(bg_data = A, by_speaker_data = por_hablante, test_speakers = hab, bg_speakers = hab)
    M <- as.matrix(r$likelihood_ratio_matrix)
    for (s in rownames(M)) for (o in colnames(M))
      filas[[length(filas) + 1]] <- data.frame(condicion = cn, vocal = v, sus = s, off = o, logLR = M[s, o], mismo = as.integer(s == o))
    cat(sprintf("%s /%s/  Cllr=%.3f EER=%.3f  (%.0f s)\n", cn, v, r$cllr, r$eer, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }
}
out <- do.call(rbind, filas)
write.csv(out, "resultados/mvkd_pares.csv", row.names = FALSE)
cat("Guardado resultados/mvkd_pares.csv:", nrow(out), "filas\n")
