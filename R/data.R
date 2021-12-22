#' Microbiome, HIV infection and MSM factor
#'
#' A dataset containing the number of counts of 60 different genera in a group
#' of 155 samples (including HIV - infected and non - infected patients).
#' The \code{data.frame} is composed by 60 genera and two variables.
#'
#' @format The \code{data.frame} is composed by 60 genera and 2 variables
#' \describe{
#'   \item{genera}{The first 60 columns, from \emph{g_Prevotella} until
#'        \emph{o_NB1-n_g_unclassified} referred to different genera.}
#'   \item{MSM}{a factor determining if the individual is \code{MSM} (\emph{Men Sex with
#'    Men}) or not (\code{nonMSM}).}
#'   \item{HIV_Status}{a factor specifying if the individual is infected
#'    (\code{Pos}) or not (\code{Neg}).}
#'
#' }
#' @docType data
#' @name HIV
#' @references \url{https://pubmed.ncbi.nlm.nih.gov/27077120/}
#' @keywords data
NULL


#' Microbiome and sCD14 inflammation parameter
#'
#' A dataset containing the number of counts of 60 different genera in a group
#' of 151 samples (including HIV - infected and non - infected patients).
#' The \code{data.frame} is composed by 60 genera and a numeric variable
#'
#' @format The \code{data.frame} is composed by 60 genera and a variable
#' \describe{
#'   \item{genera}{The first 60 columns, from \emph{g_Prevotella} until
#'   \emph{o_NB1-n_g_unclassified} referred to different genera.}
#'   \item{sCD14}{a \code{numeric} variable with the value of the inflammation
#'   parameter sCD14 for each sample.}
#' }
#' @name sCD14
#' @docType data
#' @references \doi{10.1016/j.ebiom.2016.01.032}
#' @keywords data
NULL



#' Microbiome composition related to Crohn`s disease study
#'
#' A dataset containing the number of counts of 48 different genera in a group
#' of 975 samples (including 662 samples of patients with Crohn`s disease and
#' 313 controls).
#' The \code{data.frame} is composed by 48 genera and a factor variable
#'
#' @format The \code{data.frame} is composed by 48 genera and a variable
#' \describe{
#'   \item{genera}{The first 48 columns, from \emph{g_Turicibacter} until
#'   \emph{g_Bilophila} referred to different genera.}
#'   \item{y}{a \code{factor} indicating if the sample corresponds to a case (
#'   \emph{CD}) or a control (\emph{no}).}
#' }
#' @name Crohn
#' @docType data
#' @references \url{https://qiita.ucsd.edu/}
#' @keywords data
NULL
