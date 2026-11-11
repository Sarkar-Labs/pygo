package tensor

import "fmt"

func Fused1539Log2MatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Sigmoid()
	return r, nil
}
func Fused1539Log2MatmulSigmoidName() string { return "log2_matmul_sigmoid" }
