package tensor

import "fmt"

func Fused0909FloorMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Sigmoid()
	return r, nil
}
func Fused0909FloorMatmulSigmoidName() string { return "floor_matmul_sigmoid" }
