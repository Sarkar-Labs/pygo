package tensor

import "fmt"

func Fused0263SigmoidDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Log2()
	return r, nil
}
func Fused0263SigmoidDivLog2Name() string { return "sigmoid_div_log2" }
