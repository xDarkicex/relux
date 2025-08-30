package loss

type Loss interface {
	Name() string
	Forward(yPred, yTrue []float64) float64
	Backward(yPred, yTrue []float64) []float64
}

type mse struct{}
type bce struct{}

func MSE() Loss { return mse{} }
func BCE() Loss { return bce{} }

func (mse) Name() string { return "mse" }
func (bce) Name() string { return "bce" }
