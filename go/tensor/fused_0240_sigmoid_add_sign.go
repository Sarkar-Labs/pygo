package tensor

import "fmt"

func Fused0240SigmoidAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Sign()
	return r, nil
}
func Fused0240SigmoidAddSignName() string { return "sigmoid_add_sign" }
