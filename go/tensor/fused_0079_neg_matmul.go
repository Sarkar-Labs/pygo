package tensor

import "fmt"

func Fused0079NegMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	return r, nil
}
func Fused0079NegMatmulName() string { return "neg_matmul" }
