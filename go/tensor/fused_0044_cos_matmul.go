package tensor

import "fmt"

func Fused0044CosMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	return r, nil
}
func Fused0044CosMatmulName() string { return "cos_matmul" }
