package tensor

import "fmt"

func Fused0870CosAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Sign()
	return r, nil
}
func Fused0870CosAddSignName() string { return "cos_add_sign" }
