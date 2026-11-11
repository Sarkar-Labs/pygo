package tensor

import "fmt"

func Fused0150ReluAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Sign()
	return r, nil
}
func Fused0150ReluAddSignName() string { return "relu_add_sign" }
